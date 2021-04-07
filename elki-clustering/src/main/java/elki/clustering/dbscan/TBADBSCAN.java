package elki.clustering.dbscan;

import elki.Algorithm;
import elki.clustering.ClusteringAlgorithm;
import elki.clustering.kmeans.initialization.KMeansPlusPlus;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.model.ClusterModel;
import elki.data.model.Model;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.datastore.memory.MapIntegerDBIDIntegerStore;
import elki.database.ids.*;
import elki.database.query.distance.DistanceQuery;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.distance.minkowski.EuclideanDistance;
import elki.logging.Logging;
import elki.logging.progress.FiniteProgress;
import elki.logging.progress.IndefiniteProgress;
import elki.result.Metadata;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.DoubleParameter;
import elki.utilities.optionhandling.parameters.EnumParameter;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;
import elki.utilities.random.RandomFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TBADBSCAN<O> implements ClusteringAlgorithm<Clustering<Model>> {
  private static final Logging LOG = Logging.getLogger(TBADBSCAN.class);
  protected Distance<? super O> distance;
  protected double epsilon;
  protected int minpts;
  protected int nRefPoints;
  public enum RefPointMode {
    RANDOM,
    KPP,
    QUANTIL,
  }
  protected RefPointMode refPointMode = RefPointMode.KPP;

  public TBADBSCAN(Distance<? super O> distance, double epsilon, int minpts, int nRefPoints, RefPointMode mode){
    super();
    this.distance = distance;
    this.epsilon = epsilon;
    this.minpts = minpts;
    this.nRefPoints = nRefPoints;
    this.refPointMode = mode;
  }

  @Override
  public TypeInformation[] getInputTypeRestriction() {
    return TypeUtil.array(distance.getInputTypeRestriction());
  }

  public Clustering<Model> run(Relation<O> relation){
    final int datasetSize = relation.size();
    if(datasetSize < minpts) {
      Clustering<Model> result = new Clustering<>();
      Metadata.of(result).setLongName("TBA-DBSCAN Clustering");
      result.addToplevelCluster(new Cluster<>(relation.getDBIDs(), true, ClusterModel.CLUSTER));
      return result;
    }

    Instance tbaDBSCAN = new Instance();
    tbaDBSCAN.run(relation);

    /*double averagen = tbaDBSCAN.ncounter / (double) relation.size();
    LOG.statistics(new DoubleStatistic(DBSCAN.class.getName() + ".average-neighbors", averagen));
    if(averagen < 1 + 0.1 * (minpts - 1)) {
      LOG.warning("There are very few neighbors found. Epsilon may be too small.");
    }
    if(averagen > 100 * minpts) {
      LOG.warning("There are very many neighbors found. Epsilon may be too large.");
    }*/

    Clustering<Model> result = new Clustering<>();
    Metadata.of(result).setLongName("TBA-DBSCAN Clustering");
    for(ModifiableDBIDs res: tbaDBSCAN.resultList) {
      result.addToplevelCluster(new Cluster<>(res, ClusterModel.CLUSTER));
    }
    result.addToplevelCluster(new Cluster<>(tbaDBSCAN.noise, true, ClusterModel.CLUSTER));
    return result;
  }

  private class Instance {
    protected List<ModifiableDBIDs> resultList;
    protected ModifiableDBIDs noise;
    protected ModifiableDBIDs processedIDs;
    protected FiniteProgress objprog;
    protected IndefiniteProgress clusprog;
    protected DistanceQuery<? super O> distanceQuery;
    protected ModifiableDoubleDBIDList[] refDists;
    protected MapIntegerDBIDIntegerStore[] refDistsOffsetMap;
    protected int nDistCalcs;


    protected void run(Relation<O> relation){
      final int size = relation.size();
      this.objprog = LOG.isVerbose() ? new FiniteProgress("Processing objects", size, LOG) : null;
      this.clusprog = LOG.isVerbose() ? new IndefiniteProgress("Number of clusters", LOG) : null;
      this.distanceQuery = distance.instantiate(relation);

      // Instantiate necessary objects
      resultList = new ArrayList<>();
      noise = DBIDUtil.newHashSet();
      processedIDs = DBIDUtil.newHashSet(size);
      nDistCalcs = 0;

      //RefDistances
      refDists = new ModifiableDoubleDBIDList[nRefPoints];
      refDistsOffsetMap = new MapIntegerDBIDIntegerStore[nRefPoints];
      for(int i=0; i<nRefPoints; i++){
        refDists[i] = DBIDUtil.newDistanceDBIDList(relation.size());
        refDistsOffsetMap[i] = new MapIntegerDBIDIntegerStore(relation.size());
      }

      // Calc ref points
      switch (refPointMode){
        case KPP:
          generateKMPPRefPoints(relation);
          break;
        case RANDOM:
          generateRandomRefPoints(relation);
          break;
        case QUANTIL:
          generateQuantilRefPoints(relation);
          break;
      }
      LOG.warning("Ref Mode: "+refPointMode);
      LOG.warning("Ref Dist Calcs"+nDistCalcs);

      long clustStart = System.currentTimeMillis();
      //Start Clustering here
      for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()){
        //cluster unprocessed data
        if(!processedIDs.contains(iditer)){
          expandCluster(iditer, relation);
        }

        //Update Metadata
        if(objprog != null && clusprog != null) {
          objprog.setProcessed(processedIDs.size(), LOG);
          clusprog.setProcessed(resultList.size(), LOG);
        }
        if(processedIDs.size() == size) {
          break;
        }
      }
      long clustTime = System.currentTimeMillis() - clustStart;
      LOG.warning("Ref Calc "+clustTime+" ms");
      LOG.warning("Distance Calculations "+nDistCalcs);
      // Finish progress logging
      LOG.ensureCompleted(objprog);
      LOG.setCompleted(clusprog);
    }

    protected void expandCluster(DBIDRef startObjectID, Relation<O> relation){
      ArrayModifiableDBIDs seeds = DBIDUtil.newArray();
      ModifiableDBIDs neighbors = getNeighbors(startObjectID);

      processedIDs.add(startObjectID);
      LOG.incrementProcessed(objprog);

      //No core continue
      if(neighbors.size() < minpts){
        noise.add(startObjectID);
        return;
      }

      ModifiableDBIDs currentCluster = DBIDUtil.newArray(neighbors.size());
      currentCluster.add(startObjectID);
      processNeighbors(neighbors, currentCluster, seeds);

      DBIDVar o = DBIDUtil.newVar();
      while (!seeds.isEmpty()){
        ModifiableDBIDs curSeeds = getNeighbors(seeds.pop(o));
        if (curSeeds.size() >= minpts){
          processNeighbors(curSeeds, currentCluster, seeds);
        }
        LOG.incrementProcessed(objprog);
      }
      resultList.add(currentCluster);
      LOG.incrementProcessed(clusprog);
    }

    private void processNeighbors(ModifiableDBIDs neighbors, ModifiableDBIDs currentCluster, ArrayModifiableDBIDs seeds) {
      for(DBIDIter neighbor = neighbors.iter(); neighbor.valid(); neighbor.advance()){
        if(processedIDs.add(neighbor)){
          if(!seeds.contains(neighbor)) {
            seeds.add(neighbor);
          }
        } else if(!noise.remove(neighbor)){
          continue;
        }
        currentCluster.add(neighbor);
      }
    }

    protected ModifiableDBIDs getCombinedNeighborhoodCandidates(DBIDRef point){
      ModifiableDBIDs neighboorhoodCandidates = getNeighborhoodCandidates(0, point);
      if(neighboorhoodCandidates.size() < minpts) {
        return DBIDUtil.newArray();
      }
      if (nRefPoints > 0) {
        for (int i = 1; i < nRefPoints; i++) {
          ModifiableDBIDs nextNeighboorhoodCandidates = getNeighborhoodCandidates(i, point);
          if (nextNeighboorhoodCandidates.size() < minpts) {
            return DBIDUtil.newArray();
          }
          neighboorhoodCandidates = DBIDUtil.intersection(neighboorhoodCandidates, nextNeighboorhoodCandidates);
          if (neighboorhoodCandidates.size() < minpts) {
            return DBIDUtil.newArray();
          }
        }
      }
      return neighboorhoodCandidates;
    }

    protected ModifiableDBIDs getNeighbors(DBIDRef point){
      ModifiableDBIDs neighborhoodCandidates = getCombinedNeighborhoodCandidates(point);
      ModifiableDBIDs neighbors = DBIDUtil.newArray();
      for(DBIDMIter neighborIter = neighborhoodCandidates.iter(); neighborIter.valid(); neighborIter.advance()){
        nDistCalcs++;
        if(distanceQuery.distance(point, neighborIter) <= epsilon){
          neighbors.add(neighborIter);
        }
      }
      return neighbors;
    }

    protected ModifiableDBIDs getNeighborhoodCandidates(int refPointIndex, DBIDRef point){
      int offset = refDistsOffsetMap[refPointIndex].intValue(point);
      ArrayModifiableDBIDs forwardCandidates = getForwardCandidates(refPointIndex, offset);
      ArrayModifiableDBIDs backwardCandidates = getBackwardCandidates(refPointIndex, offset);
      forwardCandidates.addDBIDs(backwardCandidates);
      return forwardCandidates;
    }

    protected ArrayModifiableDBIDs getForwardCandidates(int refPointIndex, int referencePointOffset){
      ArrayModifiableDBIDs forwardCandidates = DBIDUtil.newArray();
      double startDist = refDists[refPointIndex].doubleValue(referencePointOffset);
      double forwardThreshold = startDist + epsilon;
      for(DoubleDBIDListIter distIter = refDists[refPointIndex].iter().seek(referencePointOffset); distIter.valid(); distIter.advance()){
        if(distIter.doubleValue() > forwardThreshold){
          break;
        }
        forwardCandidates.add(distIter);
      }
      return forwardCandidates;
    }

    protected ArrayModifiableDBIDs getBackwardCandidates(int refPointIndex, int referencePointOffset){
      ArrayModifiableDBIDs backwardCandidates = DBIDUtil.newArray();
      double startDist = refDists[refPointIndex].doubleValue(referencePointOffset);
      double backwardThreshold = startDist - epsilon;
      for(DoubleDBIDListIter distIter = refDists[refPointIndex].iter().seek(referencePointOffset); distIter.valid(); distIter.retract()){
        if(distIter.doubleValue() < backwardThreshold){
          break;
        }
        backwardCandidates.add(distIter);
      }
      return backwardCandidates;
    }

    protected ArrayModifiableDBIDs getBackForwardCandidates(int refPointIndex, int referencePointOffset){
      ArrayModifiableDBIDs neighborCandidates = DBIDUtil.newArray();
      double startDistance = refDists[refPointIndex].doubleValue(referencePointOffset);
      double backwardThreshold = startDistance - epsilon;
      double forwardThreshold = startDistance + epsilon;
      int candidatesFirstIndex = -1;
      int candidatesLastIndex = -1;
      DoubleDBIDListIter candidatesListIter = refDists[refPointIndex].iter();
      for(candidatesListIter.seek(referencePointOffset); candidatesListIter.valid(); candidatesListIter.retract()){
        if(candidatesListIter.doubleValue() < backwardThreshold){
          candidatesFirstIndex = candidatesListIter.getOffset() + 1;
          break;
        }
        neighborCandidates.add(candidatesListIter);
      }
      for(candidatesListIter.seek(referencePointOffset); candidatesListIter.valid(); candidatesListIter.advance()){
        if(candidatesListIter.doubleValue() > forwardThreshold){
          candidatesLastIndex = candidatesListIter.getOffset() - 1;
          break;
        }
        neighborCandidates.add(candidatesListIter);
      }
      return neighborCandidates;
    }

    protected void getNoiseFromRefDist(int refDistIndex){
      for(int i=0; i<refDists[refDistIndex].size(); i++){
        DBIDVar point = DBIDUtil.newVar();
        refDists[refDistIndex].assignVar(i, point);
        ModifiableDBIDs neighborhoodCandidates = getNeighborhoodCandidates(refDistIndex, point);
        if(neighborhoodCandidates.size() < minpts){
          processedIDs.add(point);
          noise.add(point);
        }
      }
    }

    protected void generateSortedReferenceDistances(int index, DBIDRef refPoint, Relation<O> relation){
      for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()){
        nDistCalcs++;
        double dist = distanceQuery.distance(iditer, refPoint);
        refDists[index].add(dist, iditer);
      }
      refDists[index].sort();
    }

    protected void generateOffsetMap(int index){
      for(DoubleDBIDListIter idIter = refDists[index].iter(); idIter.valid(); idIter.advance()){
        refDistsOffsetMap[index].putInt(idIter, idIter.getOffset());
      }
    }

    protected void generateQuantilRefPoints(Relation<O> relation){
      DBIDVar startRefPoint = DBIDUtil.newVar();
      DBIDUtil.randomSample(relation.getDBIDs(), 1, new Random()).pop(startRefPoint);
      generateSortedReferenceDistances(0,startRefPoint, relation);
      generateOffsetMap(0);

      DBIDVar nextRefPoint = DBIDUtil.newVar();
      for(int i=1; i<nRefPoints; i++){
        int qunatilIndex = (int) Math.floor(refDists[i-1].size() * 0.75);
        refDists[i-1].assignVar(qunatilIndex, nextRefPoint);
        generateSortedReferenceDistances(i, nextRefPoint, relation);
        generateOffsetMap(i);
      }
    }

    protected void generateRandomRefPoints(Relation<O> relation){
      ModifiableDBIDs refPoints = DBIDUtil.randomSample(relation.getDBIDs(), nRefPoints, new Random());
      DBIDVar refPoint = DBIDUtil.newVar();
      for(int i=0; i < nRefPoints; i++){
        refPoints.pop(refPoint);
        generateSortedReferenceDistances(i, refPoint, relation);
        generateOffsetMap(i);
      }
    }

    protected void generateKMPPRefPoints(Relation<O> relation){
      KMeansPlusPlus kpp = new KMeansPlusPlus<Model>(new RandomFactory(55));
      DBIDs refPoints = kpp.chooseInitialMedoids(nRefPoints, relation.getDBIDs(), distanceQuery);
      int i = 0;
      for(DBIDIter iter = refPoints.iter(); iter.valid(); iter.advance()){
        generateSortedReferenceDistances(i, iter, relation);
        generateOffsetMap(i);
        i++;
      }
    }

  }


  /**
   * Parameterization class.
   *
   * @author Felix Krause
   */
  public static class Par<O> implements Parameterizer {
    /**
     * Parameter to specify the maximum radius of the neighborhood to be
     * considered, must be suitable to the distance function specified.
     */
    public static final OptionID EPSILON_ID = new OptionID("tbadbscan.epsilon", "The maximum radius of the neighborhood to be considered.");

    /**
     * Parameter to specify the threshold for minimum number of points in the
     * epsilon-neighborhood of a point, must be an integer greater than 0.
     */
    public static final OptionID MINPTS_ID = new OptionID("tbadbscan.minpts", "Threshold for minimum number of points in the epsilon-neighborhood of a point. The suggested value is '2 * dim - 1'.");

    /**
     * Parameter to specify the number of reference points to use for clustering.
     * Must be an integer greater than 0.
     */
    public static final OptionID NREFPOINTS_ID = new OptionID("tbadbscan.nRefPoints", "The number of reference points to use for clustering.");

    /**
     *
     */
    public static final OptionID MODE_ID = new OptionID("tbadbscan.refPointMode", "The mode of which the refpoints are chosen.");

    /**
     * Holds the epsilon radius threshold.
     */
    protected double epsilon;

    /**
     * Holds the minimum cluster size.
     */
    protected int minpts;

    /**
     * Holds the number of reference points.
     */
    protected int nRefPoints;

    protected RefPointMode mode = RefPointMode.KPP;

    /**
     * The distance function to use.
     */
    protected Distance<? super O> distance;

    @Override
    public void configure(Parameterization config) {
      new ObjectParameter<Distance<? super O>>(Algorithm.Utils.DISTANCE_FUNCTION_ID, Distance.class, EuclideanDistance.class) //
          .grab(config, x -> distance = x);
      new DoubleParameter(EPSILON_ID) //
          .addConstraint(CommonConstraints.GREATER_THAN_ZERO_DOUBLE) //
          .grab(config, x -> epsilon = x);
      if(new IntParameter(MINPTS_ID) //
          .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
          .grab(config, x -> minpts = x) && minpts <= 2) {
        LOG.warning("DBSCAN with minPts <= 2 is equivalent to single-link clustering at a single height. Consider using larger values of minPts.");
      }
      new IntParameter(NREFPOINTS_ID) //
          .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT)
          .grab(config, x -> nRefPoints = x);
      new EnumParameter<RefPointMode>(MODE_ID, RefPointMode.class, RefPointMode.KPP) //
          .grab(config, x -> mode = x);
    }

    @Override
    public TBADBSCAN<O> make() {
      return new TBADBSCAN<>(distance, epsilon, minpts, nRefPoints, mode);
    }
  }
}
