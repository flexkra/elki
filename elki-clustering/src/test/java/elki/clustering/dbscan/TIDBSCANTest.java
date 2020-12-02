package elki.clustering.dbscan;

import elki.clustering.AbstractClusterAlgorithmTest;
import elki.data.Clustering;
import elki.data.DoubleVector;
import elki.data.model.Model;
import elki.database.Database;
import elki.utilities.ELKIBuilder;
import org.junit.Test;

public class TIDBSCANTest extends AbstractClusterAlgorithmTest {
  /**
   * Run TIDBSCAN with fixed parameters and compare the result to a golden
   * standard.
   */
  @Test
  public void testTIDBSCANResults() {
    Database db = makeSimpleDatabase(UNITTEST + "3clusters-and-noise-2d.csv", 330);
    Clustering<Model> result = new ELKIBuilder<TIDBSCAN<DoubleVector>>(TIDBSCAN.class) //
        .with(TIDBSCAN.Par.EPSILON_ID, 0.04) //
        .with(TIDBSCAN.Par.MINPTS_ID, 20) //
        .build().autorun(db);
    assertFMeasure(db, result, 0.996413);
    assertClusterSizes(result, new int[] { 29, 50, 101, 150 });
  }

  /**
   * Run TIDBSCAN with fixed parameters and compare the result to a golden
   * standard.
   */
  @Test
  public void testDBSCANOnSingleLinkDataset() {
    Database db = makeSimpleDatabase(UNITTEST + "single-link-effect.ascii", 638);
    Clustering<Model> result = new ELKIBuilder<TIDBSCAN<DoubleVector>>(TIDBSCAN.class) //
        .with(TIDBSCAN.Par.EPSILON_ID, 11.5) //
        .with(TIDBSCAN.Par.MINPTS_ID, 120) //
        .build().autorun(db);
    assertFMeasure(db, result, 0.9588973429470581);
    assertClusterSizes(result, new int[] { 16, 199, 200, 223 });
  }
}
