package elki.clustering.dbscan;

import elki.Algorithm;
import elki.clustering.ClusteringAlgorithm;
import elki.clustering.kmeans.initialization.KMeansPlusPlus;
import elki.clustering.kmedoids.initialization.BUILD;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.model.ClusterModel;
import elki.data.model.Model;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.datastore.memory.MapIntegerDBIDIntegerStore;
import elki.database.ids.*;
import elki.database.query.QueryBuilder;
import elki.database.query.distance.DistanceQuery;
import elki.database.query.range.RangeSearcher;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.distance.minkowski.EuclideanDistance;
import elki.logging.Logging;
import elki.logging.progress.FiniteProgress;
import elki.logging.progress.IndefiniteProgress;
import elki.logging.statistics.LongStatistic;
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

import java.util.*;

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
    PAM,
  }
  public enum NeighborSearchMode {
    OLD,
    NEW,
  }
  protected RefPointMode refPointMode = RefPointMode.KPP;
  protected NeighborSearchMode neighborSearchMode = NeighborSearchMode.NEW;

  public TBADBSCAN(Distance<? super O> distance, double epsilon, int minpts, int nRefPoints, RefPointMode mode, NeighborSearchMode searchMode){
    super();
    this.distance = distance;
    this.epsilon = epsilon;
    this.minpts = minpts;
    this.nRefPoints = nRefPoints;
    this.refPointMode = mode;
    this.neighborSearchMode = searchMode;
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
    protected RangeSearcher<DBIDRef> rangeSearcher;
    protected ModifiableDoubleDBIDList[] refDists;
    protected MapIntegerDBIDIntegerStore[] refDistsOffsetMap;
    protected long nDistCalcs;



    // ALGORITHM

    protected void run(Relation<O> relation){
      final int size = relation.size();
      this.objprog = LOG.isVerbose() ? new FiniteProgress("Processing objects", size, LOG) : null;
      this.clusprog = LOG.isVerbose() ? new IndefiniteProgress("Number of clusters", LOG) : null;
      this.distanceQuery = distance.instantiate(relation);
      this.rangeSearcher = new QueryBuilder<>(relation, distance).rangeByDBID(epsilon);

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
        case PAM:
          generatePAMRefPoints(relation);
          break;
      }
      LOG.statistics(new LongStatistic(TBADBSCAN.class.getName() + ".initialization-distance-computations", nDistCalcs));

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
      LOG.statistics(new LongStatistic(TBADBSCAN.class.getName() + ".distance-computations", nDistCalcs));
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


    // NEIGHBOR FINDING

    protected ModifiableDBIDs getNeighbors(DBIDRef point){
      ModifiableDBIDs neighborhoodCandidates = DBIDUtil.newArray();
      switch (neighborSearchMode){
        case NEW:
          neighborhoodCandidates = getNeighborhoodCandidatesNew(point);
          break;
        case OLD:
          neighborhoodCandidates = getCombinedNeighborhoodCandidates(point);
          break;
      }

      ModifiableDBIDs neighbors = DBIDUtil.newArray();
      for(DBIDMIter neighborIter = neighborhoodCandidates.iter(); neighborIter.valid(); neighborIter.advance()){
        nDistCalcs++;
        if(distanceQuery.distance(point, neighborIter) <= epsilon){
          neighbors.add(neighborIter);
        }
      }
      return neighbors;
    }

    // OLD START
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

    // OLD END

    // NEW START

    protected ModifiableDBIDs getNeighborhoodCandidatesNew(DBIDRef point){
      //Init Neighborhoodsearch
      // lower and upper bound of neighborhood
      int[] bounds = new int[nRefPoints*2];
      // difference of upper - lower bound
      int[] neighborhoodSize = new int[nRefPoints];
      // helper array to find smallest neighborhood
      Integer[] neighborhoodSizeIndex = new Integer[nRefPoints];
      for(int x = 0; x < neighborhoodSizeIndex.length; x++){
        neighborhoodSizeIndex[x] = x;
      }

      //Getting neighborhood index bounds
      for(int i = 0; i < nRefPoints; i++){
        int currentOffset = refDistsOffsetMap[i].intValue(point);
        double currentDistValue = refDists[i].doubleValue(currentOffset);
        bounds[i*2] = newBackwardSearch(i, currentOffset);
        bounds[i*2+1] = newForwardSearch(i, currentOffset);
        // bounds[i*2+1] = binaryThresholdSearch(i, currentOffset, currentDistValue + epsilon, true);
        neighborhoodSize[i] = bounds[i*2+1] - bounds[i*2];
      }

      //Sorting neighborhoodsizes
      Arrays.sort(neighborhoodSizeIndex, (a, b) -> neighborhoodSize[a] - neighborhoodSize[b]);


      //Intersection cant get bigger than smallest neighborhood range.
      //if already smaller than minpts stop as it cant be a core point.
      if(neighborhoodSize[neighborhoodSizeIndex[0]] < minpts){
        return DBIDUtil.newHashSet(0);
      }
      //ArrayModifiableDBIDs neighborhoodCandidates = DBIDUtil.newArray(neighborhoodSize[neighborhoodSizeIndex[0]]);
      HashSetModifiableDBIDs neighborhoodCandidates =  DBIDUtil.newHashSet(neighborhoodSize[neighborhoodSizeIndex[0]]);
      //Intersect all neighborhood candidates starting by smallest neighboorhood range
      //stop if finished or smaller than minpts
      int neighborhoodIndex = neighborhoodSizeIndex[0];
      int upperbound = bounds[neighborhoodIndex*2+1];
      int lowerbound = bounds[neighborhoodIndex*2];
      neighborhoodCandidates = extractDBIDsFromDistanceMap(refDists[neighborhoodIndex].slice(lowerbound, upperbound));
      if(nRefPoints == 1){
        return neighborhoodCandidates;
      }
      for(int y = 1; y < nRefPoints; y++){
        neighborhoodIndex = neighborhoodSizeIndex[y];
        upperbound = bounds[neighborhoodIndex*2+1];
        lowerbound = bounds[neighborhoodIndex*2];
        HashSetModifiableDBIDs currNeighborhoodCandidates = extractDBIDsFromDistanceMap(refDists[neighborhoodIndex].slice(lowerbound, upperbound));
        //neighborhoodCandidates = intersectDBIDArrays(neighborhoodCandidates, currNeighborhoodCandidates);
        neighborhoodCandidates = (HashSetModifiableDBIDs) DBIDUtil.intersection(neighborhoodCandidates, currNeighborhoodCandidates);
        if(neighborhoodCandidates.size() < minpts){
          return DBIDUtil.newHashSet(0);
        }
      }
      return neighborhoodCandidates;
    }

    protected ArrayModifiableDBIDs intersectDBIDArrays(ArrayModifiableDBIDs a, ArrayModifiableDBIDs b){
      assert a.size() <= b.size();
      ArrayModifiableDBIDs intersectedArray = DBIDUtil.newArray(a.size());
      for(DBIDArrayMIter iter = a.iter(); iter.valid(); iter.advance()){
        if(b.contains(iter)){
          intersectedArray.add(iter);
        }
      }
      return intersectedArray;
    }

    protected HashSetModifiableDBIDs extractDBIDsFromDistanceMap(DoubleDBIDList distanceList){
      HashSetModifiableDBIDs dbids = DBIDUtil.newHashSet(distanceList.size());
      for(DoubleDBIDListIter iter = distanceList.iter(); iter.valid(); iter.advance()){
        dbids.add(iter);
      }
      return dbids;
    }

    protected int newForwardSearch(int refPointIndex, int refPointOffset) {
      final ModifiableDoubleDBIDList arr = refDists[refPointIndex];
      final int MAX_SIZE = arr.size() - 1;

      double startDistance = arr.doubleValue(refPointOffset);
      double forwardThreshold = startDistance + epsilon;

      int nextIndex = Math.min(refPointOffset + minpts, MAX_SIZE);
      while (nextIndex < MAX_SIZE && arr.doubleValue(nextIndex) < forwardThreshold) {
        nextIndex = Math.min(nextIndex + minpts, MAX_SIZE);
      }

      if (nextIndex == MAX_SIZE) {
        return MAX_SIZE;
      }

      int latest = nextIndex - minpts;
      DoubleDBIDListMIter iter = arr.iter();
      for (iter.seek(latest); iter.valid(); ){
        if (iter.doubleValue() > forwardThreshold) {
          break;
        }
        iter.advance();
      }

      //assert iter.doubleValue() <= forwardThreshold && iter.advance().doubleValue() > forwardThreshold;
      return iter.getOffset();
    }


    protected int newBackwardSearch(int refPointIndex, int refPointOffset) {
      final ModifiableDoubleDBIDList arr = refDists[refPointIndex];
      final int MIN_SIZE = 0;

      double startDistance = arr.doubleValue(refPointOffset);
      double backwardThreshold = startDistance - epsilon;

      int nextIndex = Math.max(refPointOffset - minpts, MIN_SIZE);
      while (nextIndex > MIN_SIZE && arr.doubleValue(nextIndex) > backwardThreshold) {
        nextIndex = Math.max(nextIndex - minpts, MIN_SIZE);
      }

      if (nextIndex == MIN_SIZE) {
        return MIN_SIZE;
      }

      int latest = nextIndex + minpts;
      DoubleDBIDListMIter iter = arr.iter();
      for (iter.seek(latest); iter.valid(); ){
        if (iter.doubleValue() < backwardThreshold) {
          break;
        }
        iter.retract();
      }

      //assert iter.doubleValue() <= forwardThreshold && iter.advance().doubleValue() > forwardThreshold;
      return iter.getOffset();
    }

    protected int binaryBackwardSearchNew(int refPointIndex, int refPointOffset){
      double startDistance = refDists[refPointIndex].doubleValue(refPointOffset);
      double backwardThreshold = startDistance - epsilon;
      if(refPointOffset >0 && refDists[refPointIndex].doubleValue(refPointOffset - 1) < backwardThreshold){
        return refPointOffset;
      }
      int currentIndex = Math.max(refPointOffset - minpts, 0);
      while (currentIndex > 0 && refDists[refPointIndex].doubleValue(currentIndex) >= backwardThreshold){
        currentIndex = Math.max(currentIndex - minpts, 0);
      }
      if (currentIndex == 0){
        return 0;
      }
      return stupidSearch(refPointIndex, backwardThreshold, false, currentIndex, currentIndex + minpts);
    }

    protected int binaryForwardSearchNew(int refPointIndex, int refPointOffset){
      final int maxrefDistIndex = refDists[refPointIndex].size() - 1;
      double startDistance = refDists[refPointIndex].doubleValue(refPointOffset);
      double forwardThreshold = startDistance + epsilon;
      if(refPointOffset > 0 && refDists[refPointIndex].doubleValue(refPointOffset - 1) > forwardThreshold){
        return refPointOffset;
      }
      int currentIndex = Math.min(refPointOffset + minpts, maxrefDistIndex);
      while (currentIndex < maxrefDistIndex && refDists[refPointIndex].doubleValue(currentIndex) <= forwardThreshold){
        currentIndex = Math.min(currentIndex + minpts, maxrefDistIndex);
      }
      if (currentIndex == maxrefDistIndex){
        return maxrefDistIndex;
      }
      return stupidSearch(refPointIndex, forwardThreshold, true, currentIndex - minpts, currentIndex);
    }

    protected int binaryThresholdSearch(int refPointIndex, int refPointOffset, double threshold, boolean isUpperThreshold){
      int low = -1;
      int high = -1;
      ModifiableDoubleDBIDList arr = refDists[refPointIndex];

      if (isUpperThreshold){
        low = refPointOffset;
        high = arr.size() - 1;
      } else {
        low = 0;
        high = refPointOffset;
      }

      while (low < high) {
        int mid = (low + high) / 2;

        if (arr.doubleValue(mid) == threshold){
          return mid;
        } else if (arr.doubleValue(mid) < threshold && mid != low) {
          low = mid;
        } else if (arr.doubleValue(mid) > threshold && mid != high) {
          high = mid;
        } else {
          if (isUpperThreshold) {
            high = low = low + 1;
          } else {
            high = low;
          }
        }
      }

      return low - 1;
    }

    protected int stupidSearch(int refDistIndex, double threshold, boolean isUpperThreshold, int start, int end){
      assert start < end;
      if(isUpperThreshold){
        while (start <= end){
          double dist = refDists[refDistIndex].doubleValue(start);
          if(dist <= threshold) {
            start++;
          } else {
            return start - 1;
          }
        }
        return start;
      } else {
        while (end >= start){
          if(refDists[refDistIndex].doubleValue(end) >= threshold) {
            end--;
          } else {
            return end + 1;
          }
        }
        return end;
      }
    }

    // NEW END

    // REFPOINTS FINDING START

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
      KMeansPlusPlus kpp = new KMeansPlusPlus<Model>(new RandomFactory(new Random().nextInt()));
      DBIDs refPoints = kpp.chooseInitialMedoids(nRefPoints, relation.getDBIDs(), distanceQuery);
      int i = 0;
      for(DBIDIter iter = refPoints.iter(); iter.valid(); iter.advance()){
        generateSortedReferenceDistances(i, iter, relation);
        generateOffsetMap(i);
        i++;
      }
    }

    protected void generatePAMRefPoints(Relation<O> relation){
      DBIDs pamInit = new BUILD().chooseInitialMedoids(nRefPoints, relation.getDBIDs(), distanceQuery);
      int i = 0;
      for(DBIDIter iter = pamInit.iter(); iter.valid(); iter.advance()){
        generateSortedReferenceDistances(i, iter, relation);
        generateOffsetMap(i);
        i++;
      }
    }

    // REFPOINT FINDING END

    // TEST METHODS

    protected int binaryBackwardSearch(int refPointIndex, int refPointOffset){
      double startDistance = refDists[refPointIndex].doubleValue(refPointOffset);
      double backwardThreshold = startDistance - epsilon;
      //Stop if next point is already out of threshold
      if(refPointOffset >0 && refDists[refPointIndex].doubleValue(refPointOffset - 1) < backwardThreshold){
        return refPointOffset;
      }
      int l = 0;
      int r = refPointOffset;
      int m = -1;
      while(l <= r){
        m = l + ((r - l) / 2);
        double midValue = refDists[refPointIndex].doubleValue(m);
        if(midValue == backwardThreshold){
          return m;
        } else {
          if(midValue > backwardThreshold){
            r = m - 1;
          } else {
            l = m + 1;
          }
        }
      }
      return m;
    }

    protected int binaryForwardSearch(int refPointIndex, int refPointOffset){
      double startDistance = refDists[refPointIndex].doubleValue(refPointOffset);
      double forwardThreshold = startDistance + epsilon;
      int l = refPointOffset;
      int r = refDists[refPointIndex].size() - 1;
      int m = -1;
      double midValue = -1;
      while(l <= r){
        m = l + ((r - l) / 2);
        midValue = refDists[refPointIndex].doubleValue(m);
        if(midValue == forwardThreshold){
          return m;
        } else {
          if(midValue > forwardThreshold){
            r = m - 1;
          } else {
            l = m + 1;
          }
        }
      }
      return m;
    }

    protected int binaryThresholdSearch(int refDistIndex, double threshold, boolean isUpperThreshold, int start, int end){
      int m = (start + end) / 2;
      if (end <= start){
        if(isUpperThreshold){
          // It's stupid but it works for now
          if (refDists[refDistIndex].doubleValue(start + 1) <= threshold){
            assert refDists[refDistIndex].doubleValue(start+2) > threshold;
            return start + 1;
          } else if (refDists[refDistIndex].doubleValue(start) <= threshold){
            return start;
          } else {
            assert refDists[refDistIndex].doubleValue(start-2) <= threshold;
            return start - 1;
          }
        } else {
          if (refDists[refDistIndex].doubleValue(end - 1) >= threshold){
            assert refDists[refDistIndex].doubleValue(end-2) < threshold;
            return end - 1;
          } else if (refDists[refDistIndex].doubleValue(end) >= threshold){
            return end;
          } else {
            assert refDists[refDistIndex].doubleValue(end+2) >= threshold;
            return end + 1;
          }
        }
      }
      if (refDists[refDistIndex].doubleValue(m) < threshold){
        return binaryThresholdSearch(refDistIndex, threshold, isUpperThreshold, m + 1, end);
      } else {
        return binaryThresholdSearch(refDistIndex, threshold, isUpperThreshold, start, m - 1);
      }
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

    public static final OptionID SEARCH_MODE_ID = new OptionID("tbadbscan.searchMode", "old vs new mode");

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

    protected NeighborSearchMode searchMode = NeighborSearchMode.NEW;

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
      new EnumParameter<NeighborSearchMode>(SEARCH_MODE_ID, NeighborSearchMode.class, NeighborSearchMode.OLD) //
          .grab(config, x -> searchMode = x);
    }

    @Override
    public TBADBSCAN<O> make() {
      return new TBADBSCAN<>(distance, epsilon, minpts, nRefPoints, mode, searchMode);
    }
  }
}
