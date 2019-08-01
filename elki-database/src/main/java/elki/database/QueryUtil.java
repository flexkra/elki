/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 *
 * Copyright (C) 2019
 * ELKI Development Team
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package elki.database;

import elki.data.NumberVector;
import elki.database.query.DatabaseQuery;
import elki.database.query.distance.DistanceQuery;
import elki.database.query.distance.PrimitiveDistanceQuery;
import elki.database.query.knn.KNNQuery;
import elki.database.query.knn.LinearScanDistanceKNNQuery;
import elki.database.query.knn.LinearScanEuclideanDistanceKNNQuery;
import elki.database.query.knn.LinearScanPrimitiveDistanceKNNQuery;
import elki.database.query.range.*;
import elki.database.query.rknn.RKNNQuery;
import elki.database.query.similarity.PrimitiveSimilarityQuery;
import elki.database.query.similarity.SimilarityQuery;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.distance.minkowski.EuclideanDistance;
import elki.similarity.Similarity;

/**
 * Static class with utilities related to querying a database.
 *
 * @author Erich Schubert
 * @since 0.4.0
 *
 * @opt nodefillcolor LemonChiffon
 *
 * @assoc - - - Database
 * @assoc - - - Relation
 * @has - - - DistanceQuery
 * @has - - - SimilarityQuery
 * @has - - - KNNQuery
 * @has - - - RangeQuery
 * @has - - - RKNNQuery
 */
public final class QueryUtil {
  /**
   * Private constructor. Static methods only.
   */
  private QueryUtil() {
    // Do not use.
  }

  /**
   * Get a distance query for a given distance function, automatically choosing
   * a relation.
   *
   * @param <O> Object type
   * @param database Database
   * @param distanceFunction Distance function
   * @param hints Optimizer hints
   * @return Distance query
   */
  public static <O> DistanceQuery<O> getDistanceQuery(Database database, Distance<? super O> distanceFunction, Object... hints) {
    final Relation<O> objectQuery = database.getRelation(distanceFunction.getInputTypeRestriction(), hints);
    return database.getDistanceQuery(objectQuery, distanceFunction, hints);
  }

  /**
   * Get a similarity query, automatically choosing a relation.
   *
   * @param <O> Object type
   * @param database Database
   * @param similarityFunction Similarity function
   * @param hints Optimizer hints
   * @return Similarity Query
   */
  public static <O> SimilarityQuery<O> getSimilarityQuery(Database database, Similarity<? super O> similarityFunction, Object... hints) {
    final Relation<O> objectQuery = database.getRelation(similarityFunction.getInputTypeRestriction(), hints);
    return database.getSimilarityQuery(objectQuery, similarityFunction, hints);
  }

  /**
   * Get a KNN query object for the given distance function.
   * <p>
   * An index is used when possible, but it may fall back to a linear scan.
   * <p>
   * Hints include:
   * <ul>
   * <li>Integer: maximum value for k needed</li>
   * <li>{@link DatabaseQuery#HINT_EXACT} -- no approximative indexes</li>
   * <li>{@link DatabaseQuery#HINT_OPTIMIZED_ONLY} -- no linear scans</li>
   * <li>{@link DatabaseQuery#HINT_HEAVY_USE} -- recommend optimization</li>
   * <li>{@link DatabaseQuery#HINT_SINGLE} -- discourage expensive
   * optimization</li>
   * </ul>
   *
   * @param <O> Object type
   * @param database Database
   * @param distanceFunction Distance function
   * @param hints Optimizer hints
   * @return KNN Query object
   */
  public static <O> KNNQuery<O> getKNNQuery(Database database, Distance<? super O> distanceFunction, Object... hints) {
    final Relation<O> relation = database.getRelation(distanceFunction.getInputTypeRestriction(), hints);
    final DistanceQuery<O> distanceQuery = relation.getDistanceQuery(distanceFunction, hints);
    return relation.getKNNQuery(distanceQuery, hints);
  }

  /**
   * Get a KNN query object for the given distance function.
   * <p>
   * An index is used when possible, but it may fall back to a linear scan.
   * <p>
   * Hints include:
   * <ul>
   * <li>Integer: maximum value for k needed</li>
   * <li>{@link DatabaseQuery#HINT_EXACT} -- no approximative indexes</li>
   * <li>{@link DatabaseQuery#HINT_OPTIMIZED_ONLY} -- no linear scans</li>
   * <li>{@link DatabaseQuery#HINT_HEAVY_USE} -- recommend optimization</li>
   * <li>{@link DatabaseQuery#HINT_SINGLE} -- discourage expensive
   * optimization</li>
   * </ul>
   *
   * @param relation Relation used
   * @param distanceFunction Distance function
   * @param hints Optimizer hints
   *
   * @param <O> Object type
   * @return KNN Query object
   */
  public static <O> KNNQuery<O> getKNNQuery(Relation<O> relation, Distance<? super O> distanceFunction, Object... hints) {
    final DistanceQuery<O> distanceQuery = relation.getDistanceQuery(distanceFunction, hints);
    return relation.getKNNQuery(distanceQuery, hints);
  }

  /**
   * Get a range query object for the given distance function for radius-based
   * neighbor search. (Range queries in ELKI refers to radius-based ranges, not
   * rectangular query windows.)
   * <p>
   * An index is used when possible, but it may fall back to a linear scan.
   * <p>
   * Hints include:
   * <ul>
   * <li>Range: maximum range requested</li>
   * <li>{@link DatabaseQuery#HINT_EXACT} -- no approximative indexes</li>
   * <li>{@link DatabaseQuery#HINT_OPTIMIZED_ONLY} -- no linear scans</li>
   * <li>{@link DatabaseQuery#HINT_HEAVY_USE} -- recommend optimization</li>
   * <li>{@link DatabaseQuery#HINT_SINGLE} -- discourage expensive
   * optimization</li>
   * </ul>
   *
   * @param <O> Object type
   * @param database Database
   * @param distanceFunction Distance function
   * @param hints Optimizer hints
   * @return KNN Query object
   */
  public static <O> RangeQuery<O> getRangeQuery(Database database, Distance<? super O> distanceFunction, Object... hints) {
    final Relation<O> relation = database.getRelation(distanceFunction.getInputTypeRestriction(), hints);
    final DistanceQuery<O> distanceQuery = relation.getDistanceQuery(distanceFunction, hints);
    return relation.getRangeQuery(distanceQuery, hints);
  }

  /**
   * Get a range query object for the given distance function for radius-based
   * neighbor search. (Range queries in ELKI refers to radius-based ranges, not
   * rectangular query windows.)
   * <p>
   * An index is used when possible, but it may fall back to a linear scan.
   * <p>
   * Hints include:
   * <ul>
   * <li>Range: maximum range requested</li>
   * <li>{@link DatabaseQuery#HINT_EXACT} -- no approximative indexes</li>
   * <li>{@link DatabaseQuery#HINT_OPTIMIZED_ONLY} -- no linear scans</li>
   * <li>{@link DatabaseQuery#HINT_HEAVY_USE} -- recommend optimization</li>
   * <li>{@link DatabaseQuery#HINT_SINGLE} -- discourage expensive
   * optimization</li>
   * </ul>
   *
   * @param relation Relation used
   * @param distanceFunction Distance function
   * @param hints Optimizer hints
   *
   * @param <O> Object type
   * @return KNN Query object
   */
  public static <O> RangeQuery<O> getRangeQuery(Relation<O> relation, Distance<? super O> distanceFunction, Object... hints) {
    final DistanceQuery<O> distanceQuery = relation.getDistanceQuery(distanceFunction, hints);
    return relation.getRangeQuery(distanceQuery, hints);
  }

  /**
   * Get a rKNN query object for the given distance function.
   * <p>
   * When possible, this will use an index, but it may default to an expensive
   * linear scan.
   * <p>
   * Hints include:
   * <ul>
   * <li>Integer: maximum value for k needed</li>
   * <li>{@link DatabaseQuery#HINT_EXACT} -- no approximative indexes</li>
   * <li>{@link DatabaseQuery#HINT_OPTIMIZED_ONLY} -- no linear scans</li>
   * <li>{@link DatabaseQuery#HINT_HEAVY_USE} -- recommend optimization</li>
   * <li>{@link DatabaseQuery#HINT_SINGLE} -- discourage expensive
   * optimization</li>
   * </ul>
   *
   * @param relation Relation used
   * @param distanceFunction Distance function
   * @param hints Optimizer hints
   *
   * @param <O> Object type
   * @return RKNN Query object
   */
  public static <O> RKNNQuery<O> getRKNNQuery(Relation<O> relation, Distance<? super O> distanceFunction, Object... hints) {
    final DistanceQuery<O> distanceQuery = relation.getDistanceQuery(distanceFunction, hints);
    return relation.getRKNNQuery(distanceQuery, hints);
  }

  /**
   * Get a linear scan query for the given distance query.
   *
   * @param <O> Object type
   * @param distanceQuery distance query
   * @return KNN query
   */
  @SuppressWarnings("unchecked")
  public static <O> KNNQuery<O> getLinearScanKNNQuery(DistanceQuery<O> distanceQuery) {
    // Slight optimizations of linear scans
    if(distanceQuery instanceof PrimitiveDistanceQuery) {
      final PrimitiveDistanceQuery<O> pdq = (PrimitiveDistanceQuery<O>) distanceQuery;
      if(EuclideanDistance.STATIC.equals(pdq.getDistance())) {
        final PrimitiveDistanceQuery<NumberVector> ndq = (PrimitiveDistanceQuery<NumberVector>) pdq;
        return (KNNQuery<O>) new LinearScanEuclideanDistanceKNNQuery<>(ndq);
      }
      return new LinearScanPrimitiveDistanceKNNQuery<>(pdq);
    }
    return new LinearScanDistanceKNNQuery<>(distanceQuery);
  }

  /**
   * Get a linear scan query for the given distance query.
   *
   * @param <O> Object type
   * @param distanceQuery distance query
   * @return Range query
   */
  @SuppressWarnings("unchecked")
  public static <O> RangeQuery<O> getLinearScanRangeQuery(DistanceQuery<O> distanceQuery) {
    // Slight optimizations of linear scans
    if(distanceQuery instanceof PrimitiveDistanceQuery) {
      final PrimitiveDistanceQuery<O> pdq = (PrimitiveDistanceQuery<O>) distanceQuery;
      if(EuclideanDistance.STATIC.equals(pdq.getDistance())) {
        final PrimitiveDistanceQuery<NumberVector> ndq = (PrimitiveDistanceQuery<NumberVector>) pdq;
        return (RangeQuery<O>) new LinearScanEuclideanDistanceRangeQuery<>(ndq);
      }
      return new LinearScanPrimitiveDistanceRangeQuery<>(pdq);
    }
    return new LinearScanDistanceRangeQuery<>(distanceQuery);
  }

  /**
   * Get a linear scan query for the given similarity query.
   *
   * @param <O> Object type
   * @param simQuery similarity query
   * @return Range query
   */
  public static <O> RangeQuery<O> getLinearScanSimilarityRangeQuery(SimilarityQuery<O> simQuery) {
    // Slight optimizations of linear scans
    if(simQuery instanceof PrimitiveSimilarityQuery) {
      final PrimitiveSimilarityQuery<O> pdq = (PrimitiveSimilarityQuery<O>) simQuery;
      return new LinearScanPrimitiveSimilarityRangeQuery<>(pdq);
    }
    return new LinearScanSimilarityRangeQuery<>(simQuery);
  }
}
