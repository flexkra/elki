package experimentalcode.marisa.index.xtree.util;

import de.lmu.ifi.dbs.elki.data.NumberVector;
import de.lmu.ifi.dbs.elki.data.spatial.SpatialComparable;
import de.lmu.ifi.dbs.elki.data.type.TypeUtil;
import de.lmu.ifi.dbs.elki.data.type.VectorFieldTypeInformation;
import de.lmu.ifi.dbs.elki.database.query.distance.SpatialPrimitiveDistanceQuery;
import de.lmu.ifi.dbs.elki.database.relation.Relation;
import de.lmu.ifi.dbs.elki.distance.distancefunction.AbstractPrimitiveDistanceFunction;
import de.lmu.ifi.dbs.elki.distance.distancefunction.SpatialPrimitiveDistanceFunction;
import de.lmu.ifi.dbs.elki.distance.distancevalue.DoubleDistance;

/**
 * Provides the square Euclidean distance for NumberVectors. Omitting the
 * {@link Math#sqrt(double)} operation is possible for distance rankings.
 * 
 * @author Marisa Thoma
 * @param <V> the type of NumberVector to compute the distances in between
 */
public class SquareEuclideanDistanceFunction extends AbstractPrimitiveDistanceFunction<NumberVector<?, ?>, DoubleDistance> implements SpatialPrimitiveDistanceFunction<NumberVector<?, ?>, DoubleDistance> {
  /**
   * Provides a Euclidean distance function that can compute the square
   * Euclidean distance (that is a DoubleDistance) for NumberVectors.
   */
  public SquareEuclideanDistanceFunction() {
    super();
  }

  @Override
  public DoubleDistance getDistanceFactory() {
    return DoubleDistance.FACTORY;
  }

  /**
   * Provides the square Euclidean distance between the given two vectors.
   * 
   * @return the square Euclidean distance between the given two vectors as an
   *         instance of {@link DoubleDistance DoubleDistance}.
   */
  @Override
  public DoubleDistance distance(NumberVector<?, ?> v1, NumberVector<?, ?> v2) {
    return new DoubleDistance(square_distance(v1, v2));
  }

  /**
   * Provides the square Euclidean distance between the given two vectors.
   * 
   * @return the square Euclidean distance between the given two vectors as an
   *         instance of {@link DoubleDistance DoubleDistance}.
   */
  private double square_distance(NumberVector<?, ?> v1, NumberVector<?, ?> v2) {
    if(v1.getDimensionality() != v2.getDimensionality()) {
      throw new IllegalArgumentException("Different dimensionality of NumberVectors" + "\n  first argument: " + v1.toString() + "\n  second argument: " + v2.toString());
    }
    double sqrDist = 0;
    for(int i = 1; i <= v1.getDimensionality(); i++) {
      double manhattanI = v1.getValue(i).doubleValue() - v2.getValue(i).doubleValue();
      sqrDist += manhattanI * manhattanI;
    }
    return sqrDist;
  }

  /**
   * Provides the Euclidean distance between the given two vectors.
   * 
   * @return the Euclidean distance between the given two vectors as an instance
   *         of {@link DoubleDistance DoubleDistance}.
   */
  private DoubleDistance sqrtDistance(NumberVector<?, ?> v1, NumberVector<?, ?> v2) {
    return new DoubleDistance(Math.sqrt(square_distance(v1, v2)));
  }

  /**
   * @return The square maximum distance between <code>v</code> and
   *         <code>mbr</code>.
   */
  private double square_MaxDist(SpatialComparable mbr, NumberVector<?, ?> v) {
    if(mbr.getDimensionality() != v.getDimensionality()) {
      throw new IllegalArgumentException("Different dimensionality of objects\n  " + "first argument: " + mbr.toString() + "\n  " + "second argument: " + v.toString());
    }

    double sqrDist = 0;
    for(int d = 1; d <= v.getDimensionality(); d++) {
      double value = v.getValue(d).doubleValue();
      double manhattanI, min = mbr.getMin(d), max = mbr.getMax(d);
      if(value < min)
        manhattanI = max - value;
      else if(value > max)
        manhattanI = value - min;
      else if(value - min > max - value)
        manhattanI = value - min;
      else
        manhattanI = max - value;

      sqrDist += manhattanI * manhattanI;
    }
    return sqrDist;
  }

  /**
   * @return The square maximum distance between <code>v</code> and
   *         <code>mbr</code>.
   */
  public DoubleDistance maxDist(SpatialComparable mbr, NumberVector<?, ?> v) {
    return new DoubleDistance(square_MaxDist(mbr, v));
  }

  /**
   * @return The square minimum distance between <code>v</code> and
   *         <code>mbr</code>.
   */
  private double square_MinDist(SpatialComparable mbr, NumberVector<?, ?> v) {
    if(mbr.getDimensionality() != v.getDimensionality()) {
      throw new IllegalArgumentException("Different dimensionality of objects\n  " + "first argument: " + mbr.toString() + "\n  " + "second argument: " + v.toString());
    }

    double sqrDist = 0;
    for(int d = 1; d <= v.getDimensionality(); d++) {
      double value = v.getValue(d).doubleValue();
      double r;
      if(value < mbr.getMin(d))
        r = mbr.getMin(d);
      else if(value > mbr.getMax(d))
        r = mbr.getMax(d);
      else
        // zero distance in dimension d
        continue;

      double manhattanI = value - r;
      sqrDist += manhattanI * manhattanI;
    }
    return sqrDist;
  }

  /**
   * @return The square minimum distance between <code>v</code> and
   *         <code>mbr</code>.
   */
  @Override
  public DoubleDistance minDist(SpatialComparable mbr, NumberVector<?, ?> v) {
    return new DoubleDistance(square_MinDist(mbr, v));
  }

  /**
   * @return The minimum distance between <code>v</code> and <code>mbr</code>.
   */
  private DoubleDistance sqrtMinDist(SpatialComparable mbr, NumberVector<?, ?> v) {
    return new DoubleDistance(Math.sqrt(square_MinDist(mbr, v)));
  }

  /**
   * @return The square minimum distance between <code>mbr1</code> and
   *         <code>mbr2</code>.
   */
  private double square_distance(SpatialComparable mbr1, SpatialComparable mbr2) {
    if(mbr1.getDimensionality() != mbr2.getDimensionality()) {
      throw new IllegalArgumentException("Different dimensionality of objects\n  " + "first argument: " + mbr1.toString() + "\n  " + "second argument: " + mbr2.toString());
    }

    double sqrDist = 0;
    for(int d = 1; d <= mbr1.getDimensionality(); d++) {
      double m1, m2;
      if(mbr1.getMax(d) < mbr2.getMin(d)) {
        m1 = mbr1.getMax(d);
        m2 = mbr2.getMin(d);
      }
      else if(mbr1.getMin(d) > mbr2.getMax(d)) {
        m1 = mbr1.getMin(d);
        m2 = mbr2.getMax(d);
      }
      else { // The ranges intersect
        continue;
      }
      double manhattanI = m1 - m2;
      sqrDist += manhattanI * manhattanI;
    }
    return sqrDist;
  }

  /**
   * @return The square minimum distance between <code>mbr1</code> and
   *         <code>mbr2</code>.
   */
  @Override
  public DoubleDistance mbrDist(SpatialComparable mbr1, SpatialComparable mbr2) {
    return new DoubleDistance(square_distance(mbr1, mbr2));
  }

  /**
   * @return The minimum distance between <code>mbr1</code> and
   *         <code>mbr2</code>.
   */
  private DoubleDistance sqrtDistance(SpatialComparable mbr1, SpatialComparable mbr2) {
    return new DoubleDistance(Math.sqrt(square_distance(mbr1, mbr2)));
  }

  private double square_CenterDistance(SpatialComparable mbr1, SpatialComparable mbr2) {
    if(mbr1.getDimensionality() != mbr2.getDimensionality()) {
      throw new IllegalArgumentException("Different dimensionality of objects\n  " + "first argument: " + mbr1.toString() + "\n  " + "second argument: " + mbr2.toString());
    }

    double sqrDist = 0;
    for(int d = 1; d <= mbr1.getDimensionality(); d++) {
      double c1 = mbr1.getMin(d) + mbr1.getMax(d);
      double c2 = mbr2.getMin(d) + mbr2.getMax(d);

      double manhattanI = (c1 - c2) / 2; // '/2' may be omitted as well
      sqrDist += manhattanI * manhattanI;
    }
    return sqrDist;
  }

  /** @return the square distance of the two MBRs' center points */
  @Override
  public DoubleDistance centerDistance(SpatialComparable mbr1, SpatialComparable mbr2) {
    return new DoubleDistance(square_CenterDistance(mbr1, mbr2));
  }

  /** @return the distance of the two MBRs' center points */
  private DoubleDistance sqrtCenterDistance(SpatialComparable mbr1, SpatialComparable mbr2) {
    return new DoubleDistance(Math.sqrt(square_CenterDistance(mbr1, mbr2)));
  }

  private double square_CenterDistance(SpatialComparable mbr, NumberVector<?, ?> v) {
    if(mbr.getDimensionality() != v.getDimensionality()) {
      throw new IllegalArgumentException("Different dimensionality of objects\n  " + "first argument: " + mbr.toString() + "\n  " + "second argument: " + v.toString());
    }

    double sqrDist = 0;
    for(int d = 1; d <= mbr.getDimensionality(); d++) {
      double c1 = mbr.getMin(d) + mbr.getMax(d);
      double c2 = v.getMin(d);

      double manhattanI = c1 / 2 - c2;
      sqrDist += manhattanI * manhattanI;
    }
    return sqrDist;
  }

  /** @return the square distance of mbr's center point to v*/
  private DoubleDistance centerDistance(SpatialComparable mbr, NumberVector<?, ?> v) {
    return new DoubleDistance(square_CenterDistance(mbr, v));
  }

  @Override
  public VectorFieldTypeInformation<? super NumberVector<?, ?>> getInputTypeRestriction() {
    return TypeUtil.NUMBER_VECTOR_FIELD;
  }

  @Override
  public <T extends NumberVector<?, ?>> SpatialPrimitiveDistanceQuery<T, DoubleDistance> instantiate(Relation<T> database) {
    return new SpatialPrimitiveDistanceQuery<T, DoubleDistance>(database, this);
  }
}