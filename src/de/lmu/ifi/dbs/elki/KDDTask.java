package de.lmu.ifi.dbs.elki;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import de.lmu.ifi.dbs.elki.algorithm.AbortException;
import de.lmu.ifi.dbs.elki.algorithm.Algorithm;
import de.lmu.ifi.dbs.elki.data.ClassLabel;
import de.lmu.ifi.dbs.elki.data.DatabaseObject;
import de.lmu.ifi.dbs.elki.database.AssociationID;
import de.lmu.ifi.dbs.elki.database.Database;
import de.lmu.ifi.dbs.elki.database.connection.DatabaseConnection;
import de.lmu.ifi.dbs.elki.database.connection.FileBasedDatabaseConnection;
import de.lmu.ifi.dbs.elki.logging.Logging;
import de.lmu.ifi.dbs.elki.logging.LoggingConfiguration;
import de.lmu.ifi.dbs.elki.logging.LoggingUtil;
import de.lmu.ifi.dbs.elki.normalization.Normalization;
import de.lmu.ifi.dbs.elki.result.AnnotationFromDatabase;
import de.lmu.ifi.dbs.elki.result.MultiResult;
import de.lmu.ifi.dbs.elki.result.Result;
import de.lmu.ifi.dbs.elki.result.ResultHandler;
import de.lmu.ifi.dbs.elki.result.ResultUtil;
import de.lmu.ifi.dbs.elki.result.ResultWriter;
import de.lmu.ifi.dbs.elki.utilities.ClassGenericsUtil;
import de.lmu.ifi.dbs.elki.utilities.ExceptionMessages;
import de.lmu.ifi.dbs.elki.utilities.UnableToComplyException;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.AbstractParameterizable;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.AttributeSettings;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.ClassParameter;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.Flag;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.Option;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.OptionHandler;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.OptionID;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.OptionUtil;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.ParameterException;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.Parameterizable;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.UnspecifiedParameterException;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.WrongParameterValueException;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.constraints.GlobalParameterConstraint;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.constraints.ParameterFlagGlobalConstraint;
import de.lmu.ifi.dbs.elki.utilities.pairs.Pair;

/**
 * Provides a KDDTask that can be used to perform any algorithm implementing
 * {@link Algorithm Algorithm} using any DatabaseConnection implementing
 * {@link de.lmu.ifi.dbs.elki.database.connection.DatabaseConnection
 * DatabaseConnection}.
 * 
 * @author Arthur Zimek
 * @param <O> the type of DatabaseObjects handled by this Algorithm
 */
public class KDDTask<O extends DatabaseObject> extends AbstractParameterizable {

  /**
   * Information for citation and version.
   */
  public static final String INFORMATION = "ELKI Version 0.2 (2009, July)\n\n" + "published in:\n" + "Elke Achtert, Thomas Bernecker, Hans-Peter Kriegel, Erich Schubert, Arthur Zimek:\n" + "ELKI in Time: ELKI 0.2 for the Performance Evaluation of Distance Measures for Time Series.\n" + "In Proc. 11th International Symposium on Spatial and Temporal Databases (SSTD 2009), Aalborg, Denmark, 2009.";

  /**
   * The newline string according to system.
   */
  private static final String NEWLINE = System.getProperty("line.separator");

  /**
   * Flag to obtain help-message.
   * <p>
   * Key: {@code -h}
   * </p>
   */
  private final Flag HELP_FLAG = new Flag(OptionID.HELP);

  /**
   * Flag to obtain help-message.
   * <p>
   * Key: {@code -help}
   * </p>
   */
  private final Flag HELP_LONG_FLAG = new Flag(OptionID.HELP_LONG);

  /**
   * Parameter to specify the algorithm to be applied, must extend
   * {@link de.lmu.ifi.dbs.elki.algorithm.Algorithm}.
   * <p>
   * Key: {@code -algorithm}
   * </p>
   */
  private final ClassParameter<Algorithm<O, Result>> ALGORITHM_PARAM = new ClassParameter<Algorithm<O, Result>>(OptionID.ALGORITHM, Algorithm.class);

  /**
   * Optional Parameter to specify a class to obtain a description for, must
   * extend {@link de.lmu.ifi.dbs.elki.utilities.optionhandling.Parameterizable}
   * .
   * <p>
   * Key: {@code -description}
   * </p>
   */
  private final ClassParameter<Parameterizable> DESCRIPTION_PARAM = new ClassParameter<Parameterizable>(OptionID.DESCRIPTION, Parameterizable.class, true);

  /**
   * Parameter to specify the database connection to be used, must extend
   * {@link de.lmu.ifi.dbs.elki.database.connection.DatabaseConnection}.
   * <p>
   * Key: {@code -dbc}
   * </p>
   * <p>
   * Default value: {@link FileBasedDatabaseConnection}
   * </p>
   */
  private final ClassParameter<DatabaseConnection<O>> DATABASE_CONNECTION_PARAM = new ClassParameter<DatabaseConnection<O>>(OptionID.DATABASE_CONNECTION, DatabaseConnection.class, FileBasedDatabaseConnection.class.getName());

  /**
   * Optional Parameter to specify a normalization in order to use a database
   * with normalized values.
   * <p>
   * Key: {@code -norm}
   * </p>
   */
  private final ClassParameter<Normalization<O>> NORMALIZATION_PARAM = new ClassParameter<Normalization<O>>(OptionID.NORMALIZATION, Normalization.class, true);

  /**
   * Flag to revert result to original values - invalid option if no
   * normalization has been performed.
   * <p>
   * Key: {@code -normUndo}
   * </p>
   */
  private final Flag NORMALIZATION_UNDO_FLAG = new Flag(OptionID.NORMALIZATION_UNDO);

  /**
   * Parameter to specify the result handler to be used, must extend
   * {@link ResultHandler}.
   * <p>
   * Key: {@code -resulthandler}
   * </p>
   * <p>
   * Default value: {@link ResultWriter}
   * </p>
   */
  private final ClassParameter<ResultHandler<O, Result>> RESULT_HANDLER_PARAM = new ClassParameter<ResultHandler<O, Result>>(OptionID.RESULT_HANDLER, ResultHandler.class, ResultWriter.class.getName());

  /**
   * Holds the algorithm to run.
   */
  private Algorithm<O, Result> algorithm;

  /**
   * Holds the database connection to have the algorithm run with.
   */
  private DatabaseConnection<O> databaseConnection;

  /**
   * Whether KDDTask has been properly initialized for calling the
   * {@link #run() run()}-method.
   */
  private boolean initialized = false;

  /**
   * A normalization - per default no normalization is used.
   */
  private Normalization<O> normalization = null;

  /**
   * Output handler.
   */
  private ResultHandler<O, Result> resulthandler = null;

  /**
   * Whether to undo normalization for result.
   */
  private boolean normalizationUndo = false;

  private OptionHandler helpOptionHandler;

  /**
   * Provides a KDDTask.
   */
  public KDDTask() {

    helpOptionHandler = new OptionHandler();
    helpOptionHandler.put(HELP_FLAG);
    helpOptionHandler.put(HELP_LONG_FLAG);
    helpOptionHandler.put(DESCRIPTION_PARAM);

    // parameter algorithm
    addOption(ALGORITHM_PARAM);

    // help flag
    addOption(HELP_FLAG);
    addOption(HELP_LONG_FLAG);

    // description parameter
    addOption(DESCRIPTION_PARAM);

    // parameter database connection
    addOption(DATABASE_CONNECTION_PARAM);

    // result handler
    addOption(RESULT_HANDLER_PARAM);

    // parameter normalization
    addOption(NORMALIZATION_PARAM);

    // normalization-undo flag
    addOption(NORMALIZATION_UNDO_FLAG);

    // normalization-undo depends on a defined normalization.
    GlobalParameterConstraint gpc = new ParameterFlagGlobalConstraint<String, String>(NORMALIZATION_PARAM, null, NORMALIZATION_UNDO_FLAG, true);
    optionHandler.setGlobalParameterConstraint(gpc);
  }

  /**
   * Returns a usage message with the specified message as leading line, and
   * information as provided by optionHandler. If an algorithm is specified, the
   * description of the algorithm is returned.
   * 
   * @return a usage message with the specified message as leading line, and
   *         information as provided by optionHandler
   */
  public String usage() {
    StringBuffer usage = new StringBuffer();
    usage.append(INFORMATION);
    usage.append(NEWLINE);
    
    // Collect options
    List<Pair<Parameterizable, Option<?>>> options = new ArrayList<Pair<Parameterizable, Option<?>>>();
    collectOptions(options);
    OptionUtil.formatForConsole(usage, 77, "   ", options);
    
    //TODO: cleanup:
    List<GlobalParameterConstraint> globalParameterConstraints = optionHandler.getGlobalParameterConstraints();
    if(!globalParameterConstraints.isEmpty()) {
      usage.append(NEWLINE).append("Global parameter constraints:");
      for(GlobalParameterConstraint gpc : globalParameterConstraints) {
        usage.append(NEWLINE).append(" - ");
        usage.append(gpc.getDescription());
      }
    }    
    
    return usage.toString();
  }

  /**
   * @see de.lmu.ifi.dbs.elki.utilities.optionhandling.AbstractParameterizable#setParameters(java.lang.String[])
   */
  @Override
  public String[] setParameters(String[] args) throws ParameterException {
    if(args.length == 0) {
      throw new AbortException("No options specified. Try flag -h to gain more information.");
    }
    helpOptionHandler.grabOptions(args);

    // description
    if(DESCRIPTION_PARAM.isSet()) {
      String descriptionClass = DESCRIPTION_PARAM.getValue();
      Parameterizable p;
      try {
        try {
          p = ClassGenericsUtil.instantiate(Algorithm.class, descriptionClass);
        }
        catch(UnableToComplyException e) {
          p = ClassGenericsUtil.instantiate(Parameterizable.class, descriptionClass);
        }
      }
      catch(UnableToComplyException e) {
        // FIXME: log here?
        LoggingUtil.exception(e.getMessage(), e);
        throw new WrongParameterValueException(DESCRIPTION_PARAM.getName(), descriptionClass, DESCRIPTION_PARAM.getFullDescription(), e);
      }
      throw new AbortException(OptionUtil.describeParameterizable(new StringBuffer(), p, 77, "   ").toString());
    }

    String[] remainingParameters = super.setParameters(args);

    // algorithm
    algorithm = ALGORITHM_PARAM.instantiateClass();
    addParameterizable(algorithm);
    remainingParameters = algorithm.setParameters(remainingParameters);

    // database connection
    databaseConnection = DATABASE_CONNECTION_PARAM.instantiateClass();
    addParameterizable(databaseConnection);
    remainingParameters = databaseConnection.setParameters(remainingParameters);

    // result handler
    resulthandler = RESULT_HANDLER_PARAM.instantiateClass();
    addParameterizable(resulthandler);
    remainingParameters = resulthandler.setParameters(remainingParameters);

    // normalization
    if(NORMALIZATION_PARAM.isSet()) {
      normalization = NORMALIZATION_PARAM.instantiateClass();
      normalizationUndo = NORMALIZATION_UNDO_FLAG.isSet();
      addParameterizable(normalization);
      remainingParameters = normalization.setParameters(remainingParameters);
    }

    // help
    if(HELP_FLAG.isSet() || HELP_LONG_FLAG.isSet()) {
      throw new AbortException(ExceptionMessages.USER_REQUESTED_HELP);
    }

    initialized = true;
    rememberParametersExcept(args, remainingParameters);
    return remainingParameters;
  }

  /**
  /**
   * Method to run the specified algorithm using the specified database
   * connection.
   * 
   * @return the result of the specified algorithm
   * @throws IllegalStateException if initialization has not been done properly
   *         (i.e. {@link #setParameters(String[]) setParameters(String[])} has
   *         not been called before calling this method)
   */
  public MultiResult run() throws IllegalStateException {
    if(initialized) {
      Database<O> db = databaseConnection.getDatabase(normalization);
      algorithm.run(db);
      MultiResult result;
      Result res = algorithm.getResult();

      // standard annotations from the source file
      // TODO: get them via databaseConnection!
      // adding them here will make the output writer think
      // that they were an part of the actual result.
      AnnotationFromDatabase<String, O> ar1 = new AnnotationFromDatabase<String, O>(db, AssociationID.LABEL);
      AnnotationFromDatabase<ClassLabel, O> ar2 = new AnnotationFromDatabase<ClassLabel, O>(db, AssociationID.CLASS);

      // insert standard annotations when we have a MultiResult
      if(res instanceof MultiResult) {
        result = (MultiResult) res;
        result.prependResult(ar1);
        result.prependResult(ar2);
      }
      else {
        // TODO: can we always wrap them in a MultiResult safely?
        result = new MultiResult();
        result.addResult(ar1);
        result.addResult(ar2);
        result.addResult(res);
      }

      if(result != null) {
        List<AttributeSettings> settings = getAttributeSettings();
        ResultUtil.setGlobalAssociation(result, AssociationID.META_SETTINGS, settings);
        
        if(normalizationUndo) {
          resulthandler.setNormalization(normalization);
        }
        resulthandler.processResult(db, result);
      }
      return result;
    }
    else {
      throw new IllegalStateException(KDDTask.class.getName() + " was not properly initialized. Need to set parameters first.");
    }
  }

  /**
   * Runs a KDD task accordingly to the specified parameters.
   * 
   * @param args parameter list according to description
   */
  public static void main(String[] args) {
    LoggingConfiguration.assertConfigured();
    Logging logger = Logging.getLogger(KDDTask.class);
    KDDTask<? extends DatabaseObject> kddTask = new KDDTask<DatabaseObject>();
    try {
      String[] remainingParameters = kddTask.setParameters(args);
      if(remainingParameters.length != 0) {
        logger.warning("Unnecessary parameters specified: " + Arrays.asList(remainingParameters) + "\n");
      }
      kddTask.run();
    }
    catch(AbortException e) {
      // ensure we actually show the message:
      LoggingConfiguration.setVerbose(true);
      if (kddTask.HELP_FLAG.isSet()) {
        logger.verbose(kddTask.usage());
      }
      logger.verbose(e.getMessage());
    }
    catch(UnspecifiedParameterException e) {
      LoggingConfiguration.setVerbose(true);
      logger.verbose(kddTask.usage());
      logger.warning(e.getMessage());
    }
    catch(ParameterException e) {
      // Note: the stack-trace is not included, since this exception is
      // supposedly only thrown with an already helpful message.
      if (kddTask.HELP_FLAG.isSet()) {
        LoggingConfiguration.setVerbose(true);
        logger.verbose(kddTask.usage());
      }
      logger.warning(e.getMessage(), e);
    }
    // any other exception
    catch(Exception e) {
      if (kddTask.HELP_FLAG.isSet()) {
        LoggingConfiguration.setVerbose(true);
        logger.verbose(kddTask.usage());
      }
      LoggingUtil.exception(e.getMessage(), e);
    }
  }
}