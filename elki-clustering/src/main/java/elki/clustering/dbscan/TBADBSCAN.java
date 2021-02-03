package elki.clustering.dbscan;

import com.sun.org.apache.xpath.internal.operations.Mod;
import elki.Algorithm;
import elki.clustering.ClusteringAlgorithm;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.DoubleVector;
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
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TBADBSCAN<O> implements ClusteringAlgorithm<Clustering<Model>> {
  private static final Logging LOG = Logging.getLogger(TBADBSCAN.class);
  protected Distance<? super O> distance;
  protected double epsilon;
  protected int minpts;
  protected int nRefPoints;

  public TBADBSCAN(Distance<? super O> distance, double epsilon, int minpts, int nRefPoints){
    super();
    this.distance = distance;
    this.epsilon = epsilon;
    this.minpts = minpts;
    this.nRefPoints = nRefPoints;
  }

  @Override
  public TypeInformation[] getInputTypeRestriction() {
    return TypeUtil.array(distance.getInputTypeRestriction());
  }

  public Clustering<Model> run(Relation<O> relation){
    final int datasetSize = relation.size();

    Instance tbaDBSCAN = new Instance();
    tbaDBSCAN.run(relation);

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


    protected void run(Relation<O> relation){
      // Calc ref points
      // Scan windows for noise
      // continue

      // setup meta
      final int size = relation.size();
      this.objprog = LOG.isVerbose() ? new FiniteProgress("Processing objects", size, LOG) : null;
      this.clusprog = LOG.isVerbose() ? new IndefiniteProgress("Number of clusters", LOG) : null;

      // Instantiate necessary objects
      distanceQuery = distance.instantiate(relation);
      resultList = new ArrayList<>();
      noise = DBIDUtil.newHashSet();
      processedIDs = DBIDUtil.newHashSet(size);

      //RefDistances
      refDists = new ModifiableDoubleDBIDList[nRefPoints];
      refDistsOffsetMap = new MapIntegerDBIDIntegerStore[nRefPoints];
      //RefDistances initializing
      for(int i=0; i<nRefPoints; i++){
        refDists[i] = DBIDUtil.newDistanceDBIDList(relation.size());
        refDistsOffsetMap[i] = new MapIntegerDBIDIntegerStore(relation.size());
      }

      // Calc ref points
      // TODO: Change which Points to use
      //ModifiableDBIDs refPoints = DBIDUtil.randomSample(relation.getDBIDs(), nRefPoints, 55);
      //DBIDVar refPoint = DBIDUtil.newVar();
      for(int i=0; i < nRefPoints; i++){
        //refPoints.pop(refPoint);
        DoubleVector refPoint = new DoubleVector(new double[] {0,0});
        generateSortedReferenceDistances(i, refPoint, relation);
        generateOffsetMap(i);
      }

      //Start Clustering here
      for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()){
        //cluster unprocessed data
        if(!processedIDs.contains(iditer)){
          expandCluster(iditer, relation);
        }

        //update Metadata
        if(objprog != null && clusprog != null) {
          objprog.setProcessed(processedIDs.size(), LOG);
          clusprog.setProcessed(resultList.size(), LOG);
        }
        if(processedIDs.size() == size) {
          break;
        }
      }
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
          //System.out.println("Already processed but not noise"+neighbor);
          continue;
        }
        currentCluster.add(neighbor);
      }
    }

    protected ModifiableDBIDs getCombinedNeighborhoodCandidates(DBIDRef point){
      ModifiableDBIDs neighboorhoodCandidates = getNeighborhoodCandidates(0, point);
      if (nRefPoints > 0) {
        for (int i = 1; i < nRefPoints; i++) {
          DBIDUtil.intersection(neighboorhoodCandidates, getNeighborhoodCandidates(i, point));
        }
      }
      return neighboorhoodCandidates;
    }

    protected ModifiableDBIDs getNeighbors(DBIDRef point){
      ModifiableDBIDs neighborhoodCandidates = getCombinedNeighborhoodCandidates(point);
      ModifiableDBIDs neighbors = DBIDUtil.newArray();
      for(DBIDIter neighborCandidate = neighborhoodCandidates.iter(); neighborCandidate.valid(); neighborCandidate.advance()){
        //System.out.print(relation.get(neighborCandidate));
        //System.out.print(";");
        if(distanceQuery.distance(point, neighborCandidate) <= epsilon){
          neighbors.add(neighborCandidate);
        }
      }
      //System.out.println();
      return neighbors;
    }

    protected ModifiableDBIDs getNeighborhoodCandidates(int refPointIndex, DBIDRef point){
      int offset = refDistsOffsetMap[refPointIndex].intValue(point);
      ModifiableDBIDs forwardCandidates = getForwardCandidates(refPointIndex, offset);
      ModifiableDBIDs backwardCandidates = getBackwardCandidates(refPointIndex, offset);
      return DBIDUtil.union(forwardCandidates, backwardCandidates);
      //TODO actual distance computation
    }

    protected ModifiableDBIDs getForwardCandidates(int refPointIndex, int referencePointOffset){
      ModifiableDBIDs forwardCandidates = DBIDUtil.newHashSet();
      double startDist = refDists[refPointIndex].doubleValue(referencePointOffset);
      double forwardThreshold = startDist + epsilon;
      for(DoubleDBIDListIter distIter = refDists[refPointIndex].iter().seek(referencePointOffset); distIter.valid(); distIter.advance()){
        //TODO actual distance computation
        if(distIter.doubleValue() <= forwardThreshold){
          forwardCandidates.add(distIter);
        }
      }
      return forwardCandidates;
    }

    protected ModifiableDBIDs getBackwardCandidates(int refPointIndex, int referencePointOffset){
      ModifiableDBIDs backwardCandidates = DBIDUtil.newHashSet();
      double startDist = refDists[refPointIndex].doubleValue(referencePointOffset);
      double backwardThreshold = startDist - epsilon;
      for(DoubleDBIDListIter distIter = refDists[refPointIndex].iter().seek(referencePointOffset); distIter.valid(); distIter.retract()){
        if(distIter.doubleValue() >= backwardThreshold){
          backwardCandidates.add(distIter);
        }
      }
      return backwardCandidates;
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
        double dist = distanceQuery.distance(iditer, refPoint);
        refDists[index].add(dist, iditer);
      }
      refDists[index].sort();
    }

    protected void generateSortedReferenceDistances(int index, DoubleVector refPoint, Relation<O> relation){
      for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()){
        double dist = distanceQuery.distance(iditer, (O) refPoint);
        refDists[index].add(dist, iditer);
      }
      refDists[index].sort();
    }

    protected void generateOffsetMap(int index){
      for(DoubleDBIDListIter idIter = refDists[index].iter(); idIter.valid(); idIter.advance()){
        refDistsOffsetMap[index].putInt(idIter, idIter.getOffset());
      }
    }

  }


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
    }

    @Override
    public TBADBSCAN<O> make() {
      return new TBADBSCAN<>(distance, epsilon, minpts, nRefPoints);
    }
  }
}
