package elki.clustering.dbscan;

import elki.Algorithm;
import elki.clustering.ClusteringAlgorithm;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.DoubleVector;
import elki.data.model.ClusterModel;
import elki.data.model.Model;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
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
import elki.result.Metadata;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.DoubleParameter;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;

import javax.swing.plaf.synth.SynthTextAreaUI;
import java.util.*;

/**
 * TI-DBSCAN: Clustering with DBSCAN by Means of the Triangle Inequality
 * <p>
 * Reference:
 * <p>
 * Marzena Kryszkiewicz, Piotr Lasek<br>
 * TI-DBSCAN: Clustering with DBSCAN by Means of the Triangle Inequality<br>
 * Int. Conf. on Rough Sets and Current Trends in Computing
 * <p>
 *
 * @author Felix Krause
 * @param <O> the type of Object the algorithm is applied to
 */
public class TIDBSCAN<O> implements ClusteringAlgorithm<Clustering<Model>> {

  /**
   * The logger for this class.
   */
  private static final Logging LOG = Logging.getLogger(TIDBSCAN.class);

  /**
   * Distance function used.
   */
  protected Distance<? super O> distance;

  /**
   * Holds the epsilon radius threshold.
   */
  protected double epsilon;

  /**
   * Holds minimum cluster size.
   */
  protected int minpts;

  /**
   * @param distance Distance function
   * @param epsilon Epsilon value
   * @param minpts Minpts parameter
   */
  public TIDBSCAN(Distance<? super O> distance, double epsilon, int minpts) {
    super();
    this.distance = distance;
    this.epsilon = epsilon;
    this.minpts = minpts;
  }

  @Override
  public TypeInformation[] getInputTypeRestriction() {
    return TypeUtil.array(distance.getInputTypeRestriction());
  }

  /**
   * Performs the TI-DBSCAN algorithm on the given database.
   *
   * @param relation Given database
   * @return The calculated clustering
   */
  public Clustering<Model> run(Relation<O> relation) {
    // Check if Clustering is possible
    final int size = relation.size();
    if(size < minpts) {
      Clustering<Model> result = new Clustering<>();
      Metadata.of(result).setLongName("TI-DBSCAN Clustering");
      result.addToplevelCluster(new Cluster<Model>(relation.getDBIDs(), true, ClusterModel.CLUSTER));
      return result;
    }

    // Run TI-DBSCAN instance
    Instance tidbscan = new Instance();
    tidbscan.run(relation, new QueryBuilder<>(relation, distance).rangeByDBID(epsilon));

    // Some information about the clustering

    // Creating and returning clustering result
    Clustering<Model> result = new Clustering<>();
    Metadata.of(result).setLongName("TI-DBSCAN Clustering");
    for(ModifiableDBIDs res: tidbscan.resultList) {
      result.addToplevelCluster(new Cluster<Model>(res, ClusterModel.CLUSTER));
    }
    result.addToplevelCluster(new Cluster<Model>(tidbscan.noise, true, ClusterModel.CLUSTER));
    return result;
  }


  private class Instance {
    /**
     * List of found clusters.
     */
    protected List<ModifiableDBIDs> resultList;

    /**
     * Set of Noise.
     */
    protected ModifiableDBIDs noise;

    /**
     * Set of processed ids.
     */
    protected ModifiableDBIDs processedIDs;

    /**
     * Number of neighbors.
     */
    protected long ncounter;

    /**
     * Progress for objects (may be null).
     */
    protected FiniteProgress objprog;

    /**
     * Progress for clusters (may be null).
     */
    protected IndefiniteProgress clusprog;

    /**
     * Distance Query to use.
     */
    protected DistanceQuery<? super O> distanceQuery;

    /**
     * Distances to refPoint
     */
    protected final ModifiableDoubleDBIDList refDist = DBIDUtil.newDistanceDBIDList();

    /**
     * Offset of latest point in refDist
     */
    protected int refDistOffset = -1;



    protected void run(Relation<O> relation, RangeSearcher<DBIDRef> rangeSearcher) {
      // TEST
      distanceQuery = distance.instantiate(relation);

      final int size = relation.size();
      this.objprog = LOG.isVerbose() ? new FiniteProgress("Processing objects", size, LOG) : null;
      this.clusprog = LOG.isVerbose() ? new IndefiniteProgress("Number of clusters", LOG) : null;

      resultList = new ArrayList<>();
      noise = DBIDUtil.newHashSet();
      processedIDs = DBIDUtil.newHashSet(size);

      // Calculate Reference Distances here
      DoubleVector ref = new DoubleVector(new double[] {0,0});
      for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()){
        //DBID current = DBIDUtil.deref(iditer);
        double dist = distanceQuery.distance(iditer, (O) ref);
        refDist.add(dist, iditer);
      }

      refDist.sort();

      // Start Clustering here
      for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
        // Actual clustering
        if(!processedIDs.contains(iditer)) {
          tiExpandCluster(iditer, relation);
        }

        // Update progress
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

    protected void tiExpandCluster(DBIDRef startObjectID, Relation<O> relation) {
      ArrayModifiableDBIDs seeds = DBIDUtil.newArray();
      HashSetModifiableDBIDs neighbors = tiNeighborhood(relation, startObjectID);

      processedIDs.add(startObjectID);
      LOG.incrementProcessed(objprog);
      ncounter += seeds.size();

      // startobj is no core
      if(neighbors.size() <= minpts) {
        noise.add(startObjectID);
        return;
      }

      // startobj is core
      ModifiableDBIDs currentCluster = DBIDUtil.newArray(neighbors.size());
      currentCluster.add(startObjectID);

      //process Neighbors of startobj here
      processNeighbors(neighbors, currentCluster, seeds);

      DBIDVar o = DBIDUtil.newVar();
      while(!seeds.isEmpty()) {
        HashSetModifiableDBIDs curSeeds = tiNeighborhood(relation, seeds.pop(o), seeds.size());
        if (curSeeds.size() >= minpts){ // is core
          processNeighbors(curSeeds, currentCluster, seeds);
        }
        LOG.incrementProcessed(objprog);
      }
      resultList.add(currentCluster);
      LOG.incrementProcessed(clusprog);
    }

    private void processNeighbors(HashSetModifiableDBIDs neighbors, ModifiableDBIDs currentCluster, ArrayModifiableDBIDs seeds) {
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

    protected HashSetModifiableDBIDs tiNeighborhood(Relation<O> relation, DBIDRef point) {
      HashSetModifiableDBIDs forwardSet = tiForwardNeighborhood(relation, point);
      HashSetModifiableDBIDs backwardSet = tiBackwardNeighborhood(relation, point);
      return (HashSetModifiableDBIDs) DBIDUtil.union(forwardSet, backwardSet);
    }

    protected HashSetModifiableDBIDs tiBackwardNeighborhood(Relation<O> relation, DBIDRef point) {
      HashSetModifiableDBIDs backwardSet = DBIDUtil.newHashSet();

      boolean idFound = false;
      double backwardThreshold = 0;
      for(DoubleDBIDListIter iter = refDist.iter().seek(refDist.size()-1); iter.valid(); iter.retract()){
        if(DBIDUtil.equal(iter, point)){
          idFound = true;
          refDistOffset = iter.getOffset();
          backwardThreshold = iter.doubleValue() - epsilon;
          //continue to not add point to set
          continue;
        }
        if(idFound){
          if(iter.doubleValue() < backwardThreshold){
            break;
          }
          if(distanceQuery.distance(iter, point) <= epsilon){
            backwardSet.add(iter);
          }
        }
      }
      return backwardSet;
    }

    protected HashSetModifiableDBIDs tiForwardNeighborhood(Relation<O> relation, DBIDRef point) {
      HashSetModifiableDBIDs forwardSet = DBIDUtil.newHashSet();

      boolean idFound = false;
      double forwardThreshold = 0;
      for(DoubleDBIDListIter iter = refDist.iter(); iter.valid(); iter.advance()){
        if(DBIDUtil.equal(iter, point)){
          idFound = true;
          refDistOffset = iter.getOffset();
          forwardThreshold = iter.doubleValue() + epsilon;
          //continue to not add point to set
          continue;
        }
        if(idFound){
          if(iter.doubleValue() > forwardThreshold){
            break;
          }
          if (distanceQuery.distance(iter, point) <= epsilon){
            forwardSet.add(iter);
          }
        }
      }
      return forwardSet;
    }

    protected HashSetModifiableDBIDs tiNeighborhood(Relation<O> relation, DBIDRef point, int neighborhoodSize) {
      int backwardOffset = Math.min(refDistOffset + neighborhoodSize, refDist.size() - 1);
      int forwardOffset = Math.max(refDistOffset - neighborhoodSize, 0);
      HashSetModifiableDBIDs forwardSet = tiForwardNeighborhood(relation, point, forwardOffset);
      HashSetModifiableDBIDs backwardSet = tiBackwardNeighborhood(relation, point, backwardOffset);
      return (HashSetModifiableDBIDs) DBIDUtil.union(forwardSet, backwardSet);
    }

    protected HashSetModifiableDBIDs tiBackwardNeighborhood(Relation<O> relation, DBIDRef point, int offset) {
      HashSetModifiableDBIDs backwardSet = DBIDUtil.newHashSet();

      boolean idFound = false;
      double backwardThreshold = 0;
      for(DoubleDBIDListIter iter = refDist.iter().seek(offset); iter.valid(); iter.retract()){
        if(DBIDUtil.equal(iter, point)){
          idFound = true;
          refDistOffset = iter.getOffset();
          backwardThreshold = iter.doubleValue() - epsilon;
          //continue to not add point to set
          continue;
        }
        if(idFound){
          if(iter.doubleValue() < backwardThreshold){
            break;
          }
          if(distanceQuery.distance(iter, point) <= epsilon){
            backwardSet.add(iter);
          }
        }
      }
      return backwardSet;
    }

    protected HashSetModifiableDBIDs tiForwardNeighborhood(Relation<O> relation, DBIDRef point, int offset) {
      HashSetModifiableDBIDs forwardSet = DBIDUtil.newHashSet();

      boolean idFound = false;
      double forwardThreshold = 0;
      for(DoubleDBIDListIter iter = refDist.iter().seek(offset); iter.valid(); iter.advance()){
        if(DBIDUtil.equal(iter, point)){
          idFound = true;
          refDistOffset = iter.getOffset();
          forwardThreshold = iter.doubleValue() + epsilon;
          //continue to not add point to set
          continue;
        }
        if(idFound){
          if(iter.doubleValue() > forwardThreshold){
            break;
          }
          if (distanceQuery.distance(iter, point) <= epsilon){
            forwardSet.add(iter);
          }
        }
      }
      return forwardSet;
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
    public static final OptionID EPSILON_ID = new OptionID("tidbscan.epsilon", "The maximum radius of the neighborhood to be considered.");

    /**
     * Parameter to specify the threshold for minimum number of points in the
     * epsilon-neighborhood of a point, must be an integer greater than 0.
     */
    public static final OptionID MINPTS_ID = new OptionID("tidbscan.minpts", "Threshold for minimum number of points in the epsilon-neighborhood of a point. The suggested value is '2 * dim - 1'.");

    /**
     * Holds the epsilon radius threshold.
     */
    protected double epsilon;

    /**
     * Holds the minimum cluster size.
     */
    protected int minpts;

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
    }

    @Override
    public TIDBSCAN<O> make() {
      return new TIDBSCAN<>(distance, epsilon, minpts);
    }
  }
}
