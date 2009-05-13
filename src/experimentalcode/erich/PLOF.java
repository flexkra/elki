package experimentalcode.erich;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import de.lmu.ifi.dbs.elki.algorithm.DistanceBasedAlgorithm;
import de.lmu.ifi.dbs.elki.data.DatabaseObject;
import de.lmu.ifi.dbs.elki.database.AssociationID;
import de.lmu.ifi.dbs.elki.database.Database;
import de.lmu.ifi.dbs.elki.database.DistanceResultPair;
import de.lmu.ifi.dbs.elki.distance.DoubleDistance;
import de.lmu.ifi.dbs.elki.distance.distancefunction.DistanceFunction;
import de.lmu.ifi.dbs.elki.distance.distancefunction.EuclideanDistanceFunction;
import de.lmu.ifi.dbs.elki.math.ErrorFunctions;
import de.lmu.ifi.dbs.elki.math.MeanVariance;
import de.lmu.ifi.dbs.elki.preprocessing.MaterializeKNNPreprocessor;
import de.lmu.ifi.dbs.elki.result.AnnotationFromHashMap;
import de.lmu.ifi.dbs.elki.result.MultiResult;
import de.lmu.ifi.dbs.elki.result.OrderingFromHashMap;
import de.lmu.ifi.dbs.elki.utilities.ClassGenericsUtil;
import de.lmu.ifi.dbs.elki.utilities.Description;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.ClassParameter;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.IntParameter;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.OptionID;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.OptionUtil;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.ParameterException;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.UnusedParameterException;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.constraints.GreaterConstraint;
import de.lmu.ifi.dbs.elki.utilities.progress.FiniteProgress;

/**
 * Algorithm to compute density-based <em>probabilistic</em> local outlier factors in
 * a database based on a specified parameter {@link #K_ID}.
 * 
 * @author Erich Schubert
 * @param <O> the type of DatabaseObjects handled by this Algorithm
 */
public class PLOF<O extends DatabaseObject> extends DistanceBasedAlgorithm<O, DoubleDistance, MultiResult> {
  /**
   * OptionID for {@link #REACHABILITY_DISTANCE_FUNCTION_PARAM}
   */
  public static final OptionID REACHABILITY_DISTANCE_FUNCTION_ID = OptionID.getOrCreateOptionID("lof.reachdistfunction", "Distance function to determine the reachability distance between database objects.");

  /**
   * The distance function to determine the reachability distance between
   * database objects.
   * <p>
   * Default value: {@link EuclideanDistanceFunction}
   * </p>
   * <p>
   * Key: {@code -lof.reachdistfunction}
   * </p>
   */
  private final ClassParameter<DistanceFunction<O, DoubleDistance>> REACHABILITY_DISTANCE_FUNCTION_PARAM = new ClassParameter<DistanceFunction<O, DoubleDistance>>(REACHABILITY_DISTANCE_FUNCTION_ID, DistanceFunction.class, true);

  /**
   * The association id to associate the LOF_SCORE of an object for the
   * LOF_SCORE algorithm.
   */
  public static final AssociationID<Double> PLOF_SCORE = AssociationID.getOrCreateAssociationID("plof", Double.class);

  /**
   * Holds the instance of the reachability distance function specified by
   * {@link #REACHABILITY_DISTANCE_FUNCTION_PARAM}.
   */
  private DistanceFunction<O, DoubleDistance> reachabilityDistanceFunction;

  /**
   * OptionID for {@link #K_PARAM}
   */
  public static final OptionID K_ID = OptionID.getOrCreateOptionID("lof.k", "The number of nearest neighbors of an object to be considered for computing its LOF_SCORE.");

  /**
   * Parameter to specify the number of nearest neighbors of an object to be
   * considered for computing its LOF_SCORE, must be an integer greater than 1.
   * <p>
   * Key: {@code -lof.k}
   * </p>
   */
  private final IntParameter K_PARAM = new IntParameter(K_ID, new GreaterConstraint(1));

  /**
   * Holds the value of {@link #K_PARAM}.
   */
  int k;

  /**
   * Provides the result of the algorithm.
   */
  MultiResult result;

  /**
   * Preprocessor Step 1
   */
  MaterializeKNNPreprocessor<O, DoubleDistance> preprocessor1;

  /**
   * Preprocessor Step 2
   */
  MaterializeKNNPreprocessor<O, DoubleDistance> preprocessor2;
  
  /**
   * Include object itself in kNN neighborhood.
   * 
   * In the official LOF publication, the point itself is not considered to be
   * part of its k nearest neighbors.
   */
  boolean objectIsInKNN = false;

  /**
   * Provides the Generalized LOF_SCORE algorithm, adding parameters
   * {@link #K_PARAM} and {@link #REACHABILITY_DISTANCE_FUNCTION_PARAM} to the
   * option handler additionally to parameters of super class.
   */
  public PLOF() {
    super();
    // parameter k
    addOption(K_PARAM);
    // parameter reachability distance function
    addOption(REACHABILITY_DISTANCE_FUNCTION_PARAM);
    
    preprocessor1 = new MaterializeKNNPreprocessor<O, DoubleDistance>();
    preprocessor2 = new MaterializeKNNPreprocessor<O, DoubleDistance>();
  }

  /**
   * Performs the Generalized LOF_SCORE algorithm on the given database.
   */
  @Override
  protected MultiResult runInTime(Database<O> database) throws IllegalStateException {
    final double sqrt2 = Math.sqrt(2.0);
    getDistanceFunction().setDatabase(database, isVerbose(), isTime());
    reachabilityDistanceFunction.setDatabase(database, isVerbose(), isTime());

    // materialize neighborhoods
    HashMap<Integer, List<DistanceResultPair<DoubleDistance>>> neigh1;
    HashMap<Integer, List<DistanceResultPair<DoubleDistance>>> neigh2;
    if(logger.isVerbose()) {
      logger.verbose("Materializing Neighborhoods with respect to primary distance.");
    }
    preprocessor1.run(database, isVerbose(), isTime());
    neigh1 = preprocessor1.getMaterialized();
    if (getDistanceFunction() != reachabilityDistanceFunction) {
      if(logger.isVerbose()) {
        logger.verbose("Materializing Neighborhoods with respect to reachability distance.");
      }
      preprocessor2.run(database, isVerbose(), isTime());
      neigh2 = preprocessor2.getMaterialized();
    } else {
      if(logger.isVerbose()) {
        logger.verbose("Reusing neighborhoods of primary distance.");
      }
      neigh2 = neigh1;
    }

    HashMap<Integer, Double> lrds = new HashMap<Integer, Double>();
    {// computing LRDs
      if(logger.isVerbose()) {
        logger.verbose("Computing LRDs");
      }
      FiniteProgress lrdsProgress = new FiniteProgress("LRD", database.size());
      int counter = 0;
      for(Integer id : database) {
        counter ++;
        double sum = 0;
        List<DistanceResultPair<DoubleDistance>> neighbors = neigh2.get(id);
        int nsize = neighbors.size() - (objectIsInKNN ? 0 : 1);
        for(DistanceResultPair<DoubleDistance> neighbor : neighbors) {
          if (objectIsInKNN || neighbor.getID() != id) {
            List<DistanceResultPair<DoubleDistance>> neighborsNeighbors = neigh2.get(neighbor.getID());
            sum += Math.max(neighbor.getDistance().getValue(), neighborsNeighbors.get(neighborsNeighbors.size() - 1).getDistance().getValue());
          }
        }
        Double lrd = nsize / sum;
        lrds.put(id, lrd);
        if(logger.isVerbose()) {
          lrdsProgress.setProcessed(counter);
          logger.progress(lrdsProgress);
        }
      }
    }
    // Compute final PLOF values.
    HashMap<Integer, Double> plofs = new HashMap<Integer, Double>();
    {// compute PLOF_SCORE of each db object
      if(logger.isVerbose()) {
        logger.verbose("computing PLOFs");
      }

      FiniteProgress progressLOFs = new FiniteProgress("PLOF_SCORE for objects", database.size());
      int counter = 0;
      for(Integer id : database) {
        counter ++;
        double lrdp = lrds.get(id);
        List<DistanceResultPair<DoubleDistance>> neighbors = neigh1.get(id);
        MeanVariance mv = new MeanVariance();
        for(DistanceResultPair<DoubleDistance> neighbor1 : neighbors) {
          if (objectIsInKNN || neighbor1.getID() != id) {
            double lrdo = lrds.get(neighbor1.getSecond());
            mv.addData(lrdo);
          }
        }
        double plof = Math.min(lrdp - mv.getMean(), 0.0);
        plof = ErrorFunctions.erf(plof / (mv.getStddev() * sqrt2));
        plofs.put(id, plof);
        
        if(logger.isVerbose()) {
          progressLOFs.setProcessed(counter);
          logger.progress(progressLOFs);
        }
      }
    }
    
    // Build result representation.
    result = new MultiResult();
    result.addResult(new AnnotationFromHashMap<Double>(PLOF_SCORE, plofs));
    result.addResult(new OrderingFromHashMap<Double>(plofs, true));

    return result;
  }

  public Description getDescription() {
    return new Description(
        "PLOF",
        "Probabilistic Local Outlier Factor",
        "Variant of the LOF algorithm normalized using statistical values.",
        "unpublished");
  }

  /**
   * Calls the super method and sets additionally the value of the parameter
   * {@link #K_PARAM} and instantiates {@link #reachabilityDistanceFunction}
   * according to the value of parameter
   * {@link #REACHABILITY_DISTANCE_FUNCTION_PARAM}. The remaining parameters are
   * passed to the {@link #reachabilityDistanceFunction}.
   */
  @Override
  public String[] setParameters(String[] args) throws ParameterException {
    String[] remainingParameters = super.setParameters(args);

    // k
    k = K_PARAM.getValue();

    // reachabilityDistanceFunction - for parameter handling.
    if (REACHABILITY_DISTANCE_FUNCTION_PARAM.isSet()) {
      reachabilityDistanceFunction = REACHABILITY_DISTANCE_FUNCTION_PARAM.instantiateClass();
      remainingParameters = reachabilityDistanceFunction.setParameters(remainingParameters);
      addParameterizable(reachabilityDistanceFunction);
    } else {
      reachabilityDistanceFunction = getDistanceFunction();
    }
    
    // configure first preprocessor
    List<String> preprocParams1 = new ArrayList<String>();
    OptionUtil.addParameter(preprocParams1, MaterializeKNNPreprocessor.K_ID, Integer.toString(k+(objectIsInKNN ? 0 : 1)));
    OptionUtil.addParameter(preprocParams1, MaterializeKNNPreprocessor.DISTANCE_FUNCTION_ID, getDistanceFunction().getClass().getCanonicalName());
    OptionUtil.addParameters(preprocParams1, getDistanceFunction().getParameters());
    String[] remaining1 = preprocessor1.setParameters(ClassGenericsUtil.toArray(preprocParams1, String.class));
    if (remaining1.length > 0) {
      throw new UnusedParameterException("First preprocessor did not use all parameters.");
    }

    // configure second preprocessor
    List<String> preprocParams2 = new ArrayList<String>();
    OptionUtil.addParameter(preprocParams2, MaterializeKNNPreprocessor.K_ID, Integer.toString(k+(objectIsInKNN ? 0 : 1)));
    OptionUtil.addParameter(preprocParams2, MaterializeKNNPreprocessor.DISTANCE_FUNCTION_ID, reachabilityDistanceFunction.getClass().getCanonicalName());
    OptionUtil.addParameters(preprocParams2, reachabilityDistanceFunction.getParameters());
    String[] remaining2 = preprocessor2.setParameters(ClassGenericsUtil.toArray(preprocParams2, String.class));
    if (remaining2.length > 0) {
      throw new UnusedParameterException("Second preprocessor did not use all parameters.");
    }

    rememberParametersExcept(args, remainingParameters);
    return remainingParameters;
  }

  /**
   * Calls the super method and appends the parameter description of
   * {@link #reachabilityDistanceFunction} (if it is already initialized).
   */
  @Override
  public String parameterDescription() {
    StringBuilder description = new StringBuilder();
    description.append(super.parameterDescription());

    // reachabilityDistanceFunction
    if(reachabilityDistanceFunction != null) {
      description.append(Description.NEWLINE);
      description.append(reachabilityDistanceFunction.parameterDescription());
    }

    return description.toString();
  }

  public MultiResult getResult() {
    return result;
  }
  
  
}
