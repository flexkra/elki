package de.lmu.ifi.dbs.elki.algorithm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import de.lmu.ifi.dbs.elki.data.type.TypeInformation;
import de.lmu.ifi.dbs.elki.data.type.TypeUtil;
import de.lmu.ifi.dbs.elki.database.Database;
import de.lmu.ifi.dbs.elki.database.ids.DBID;
import de.lmu.ifi.dbs.elki.database.query.DistanceResultPair;
import de.lmu.ifi.dbs.elki.database.query.distance.DistanceQuery;
import de.lmu.ifi.dbs.elki.database.query.knn.KNNQuery;
import de.lmu.ifi.dbs.elki.database.relation.Relation;
import de.lmu.ifi.dbs.elki.distance.distancefunction.DistanceFunction;
import de.lmu.ifi.dbs.elki.distance.distancevalue.Distance;
import de.lmu.ifi.dbs.elki.logging.Logging;
import de.lmu.ifi.dbs.elki.result.KNNDistanceOrderResult;
import de.lmu.ifi.dbs.elki.utilities.documentation.Description;
import de.lmu.ifi.dbs.elki.utilities.documentation.Title;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.OptionID;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.constraints.GreaterConstraint;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.constraints.IntervalConstraint;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.parameterization.Parameterization;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.parameters.DoubleParameter;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.parameters.IntParameter;

/**
 * Provides an order of the kNN-distances for all objects within the database.
 * 
 * @author Arthur Zimek
 * @param <O> the type of DatabaseObjects handled by this Algorithm
 * @param <D> the type of Distance used by this Algorithm
 */
// TODO: redundant to kNN outlier detection?
@Title("KNN-Distance-Order")
@Description("Assesses the knn distances for a specified k and orders them.")
public class KNNDistanceOrder<O, D extends Distance<D>> extends AbstractDistanceBasedAlgorithm<O, D, KNNDistanceOrderResult<D>> {
  /**
   * The logger for this class.
   */
  private static final Logging logger = Logging.getLogger(KNNDistanceOrder.class);

  /**
   * Parameter to specify the distance of the k-distant object to be assessed,
   * must be an integer greater than 0.
   */
  public static final OptionID K_ID = OptionID.getOrCreateOptionID("knndistanceorder.k", "Specifies the distance of the k-distant object to be assessed.");

  /**
   * Holds the value of {@link #K_ID}.
   */
  private int k;

  /**
   * Parameter to specify the average percentage of distances randomly chosen to
   * be provided in the result, must be a double greater than 0 and less than or
   * equal to 1.
   */
  public static final OptionID PERCENTAGE_ID = OptionID.getOrCreateOptionID("knndistanceorder.percentage", "The average percentage of distances randomly choosen to be provided in the result.");

  /**
   * Holds the value of {@link #PERCENTAGE_ID}.
   */
  private double percentage;

  /**
   * Constructor.
   * 
   * @param distanceFunction Distance function
   * @param k k Parameter
   * @param percentage percentage parameter
   */
  public KNNDistanceOrder(DistanceFunction<O, D> distanceFunction, int k, double percentage) {
    super(distanceFunction);
    this.k = k;
    this.percentage = percentage;
  }

  /**
   * Provides an order of the kNN-distances for all objects within the specified
   * database.
   */
  @Override
  public KNNDistanceOrderResult<D> run(Database database) throws IllegalStateException {
    final Relation<O> dataQuery = database.getRelation(getInputTypeRestriction()[0]);
    final DistanceQuery<O, D> distanceQuery = database.getDistanceQuery(dataQuery, getDistanceFunction());
    final KNNQuery<O, D> knnQuery = database.getKNNQuery(distanceQuery, k);

    final Random random = new Random();
    List<D> knnDistances = new ArrayList<D>();
    for(Iterator<DBID> iter = dataQuery.iterDBIDs(); iter.hasNext();) {
      DBID id = iter.next();
      if(random.nextDouble() < percentage) {
        final List<DistanceResultPair<D>> neighbors = knnQuery.getKNNForDBID(id, k);
        final int last = Math.min(k - 1, neighbors.size() - 1);
        knnDistances.add(neighbors.get(last).getDistance());
      }
    }
    Collections.sort(knnDistances, Collections.reverseOrder());
    return new KNNDistanceOrderResult<D>("kNN distance order", "knn-order", knnDistances);
  }

  @Override
  public TypeInformation[] getInputTypeRestriction() {
    return TypeUtil.array(getDistanceFunction().getInputTypeRestriction());
  }

  @Override
  protected Logging getLogger() {
    return logger;
  }

  /**
   * Parameterization class.
   * 
   * @author Erich Schubert
   * 
   * @apiviz.exclude
   */
  public static class Parameterizer<O, D extends Distance<D>> extends AbstractDistanceBasedAlgorithm.Parameterizer<O, D> {
    protected int k;

    protected double percentage;

    public Parameterizer() {
      super();
    }

    @Override
    protected void makeOptions(Parameterization config) {
      super.makeOptions(config);
      IntParameter kP = new IntParameter(K_ID, 1);
      kP.addConstraint(new GreaterConstraint(0));
      if(config.grab(kP)) {
        k = kP.getValue();
      }

      DoubleParameter percentageP = new DoubleParameter(PERCENTAGE_ID, 1.0);
      percentageP.addConstraint(new IntervalConstraint(0, IntervalConstraint.IntervalBoundary.OPEN, 1, IntervalConstraint.IntervalBoundary.CLOSE));
      if(config.grab(percentageP)) {
        percentage = percentageP.getValue();
      }
    }

    @Override
    protected KNNDistanceOrder<O, D> makeInstance() {
      return new KNNDistanceOrder<O, D>(distanceFunction, k, percentage);
    }
  }
}