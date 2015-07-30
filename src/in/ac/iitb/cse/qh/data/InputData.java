package in.ac.iitb.cse.qh.data;

import in.ac.iitb.cse.qh.marti.MBGraphBean;
import in.ac.iitb.cse.qh.marti.MBGraphvisGenerator;
import in.ac.iitb.cse.qh.marti.MartiBoost;
import in.ac.iitb.cse.qh.meta.ClassifierProxy;
import in.ac.iitb.cse.qh.meta.ConfusionMatrixLoader;
import in.ac.iitb.cse.qh.meta.MetaModelGenerator;
import in.ac.iitb.cse.qh.meta.ModelParamLoader;
import in.ac.iitb.cse.qh.meta.Optimizer;
import in.ac.iitb.cse.qh.meta.PredictionsLoader;
import in.ac.iitb.cse.qh.meta.TargetStateCalculator;
import in.ac.iitb.cse.qh.util.BeanFinder;
import in.ac.iitb.cse.qh.util.MetaConstants;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringWriter;
import java.io.Writer;
import java.util.List;

import javax.faces.bean.ManagedProperty;

public class InputData {
	/*
	 * Properties applicable for validation data
	 */
	private List<InputPredictionInstance> predInstances;
	private ConfusionMatrix confMatrix;
	private BiasMatrix biasMatrix;
	private DisplayMatrix dispMatrix;
	/*
	 * Properties applicable for training data
	 */
	private List<InputPredictionInstance> trainPredInstances;
	private ConfusionMatrix trainConfMatrix;

	private ModelParams params;
	private boolean error = false;
	private String errorLog;

	private ClassifierProxy proxy;

	private double initialStepSize = MetaConstants.INITIAL_STEP;
	private int maxIterations = MetaConstants.MAX_OPTIM_ITERATIONS;
	private String trainFile;
	private String holdoutFile;
	private int mbLevels = MetaConstants.DEFAULT_MB_LEVELS;
	private boolean blnBoost;
	@ManagedProperty(value = "#{mbgraph}")
	private MBGraphBean mbGraph;

	public boolean isBlnBoost() {
		return blnBoost;
	}

	public void setBlnBoost(boolean blnBoost) {
		this.blnBoost = blnBoost;
	}

	public int getMbLevels() {
		return mbLevels;
	}

	public void setMbLevels(int mbLevels) {
		this.mbLevels = mbLevels;
	}

	public ClassifierProxy getProxy() {
		return proxy;
	}

	public void setProxy(ClassifierProxy proxy) {
		this.proxy = proxy;
	}

	public List<InputPredictionInstance> getTrainPredInstances() {
		return trainPredInstances;
	}

	public void setTrainPredInstances(
			List<InputPredictionInstance> trainPredInstances) {
		this.trainPredInstances = trainPredInstances;
	}

	public ConfusionMatrix getTrainConfMatrix() {
		return trainConfMatrix;
	}

	public void setTrainConfMatrix(ConfusionMatrix trainConfMatrix) {
		this.trainConfMatrix = trainConfMatrix;
	}

	public void setError(boolean error) {
		this.error = error;
	}

	public void setErrorLog(Throwable e) {
		errorLog = null != e.getCause() ? e.getCause().getMessage() : e
				.getMessage();
		if (null == errorLog || "".equalsIgnoreCase(errorLog))
			errorLog = "Error loading data. Please check the server logs for error.";
	}

	public void setErrorLog(String errorLog) {
		this.errorLog = errorLog;
	}

	public String getErrorLog() {
		return errorLog;
	}

	public boolean isError() {
		return error;
	}

	public String getTrainFile() {
		return trainFile;
	}

	public void setTrainFile(String trainFile) {
		this.trainFile = MetaConstants.UPLOAD_PATH + trainFile;
	}

	public String getHoldoutFile() {
		return holdoutFile;
	}

	public void setHoldoutFile(String holdoutFile) {
		this.holdoutFile = MetaConstants.UPLOAD_PATH + holdoutFile;
	}

	public double getInitialStepSize() {
		return initialStepSize;
	}

	public void setInitialStepSize(double initialStepSize) {
		System.out.println("setting intial step size: " + initialStepSize);
		this.initialStepSize = initialStepSize;
	}

	public MBGraphBean getMbGraph() {
		return mbGraph;
	}

	public int getMaxIterations() {
		return maxIterations;
	}

	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}

	public DisplayMatrix getDispMatrix() throws Exception {
		if (null == dispMatrix)
			dispMatrix = new DisplayMatrix();
		// loadData(MetaConstants.IN_FILE_PATH);
		// loadData();
		return dispMatrix;
	}

	public void setDispMatrix(DisplayMatrix dispMatrix) {
		this.dispMatrix = dispMatrix;
	}

	public BiasMatrix getBiasMatrix() throws Exception {
		if (null == biasMatrix)
			biasMatrix = new BiasMatrix();
		// loadData(MetaConstants.IN_FILE_PATH);
		// loadData();
		return biasMatrix;
	}

	public void setBiasMatrix(BiasMatrix biasMatrix) {
		this.biasMatrix = biasMatrix;
	}

	public ModelParams getParams() {
		return params;
	}

	public void setParams(ModelParams params) {
		this.params = params;
	}

	public List<InputPredictionInstance> getPredInstances() {
		return predInstances;
	}

	public void setPredInstances(List<InputPredictionInstance> predInstances) {
		this.predInstances = predInstances;
	}

	public ConfusionMatrix getConfMatrix() throws Exception {
		if (null == confMatrix)
			confMatrix = new ConfusionMatrix();
		// loadData(MetaConstants.IN_FILE_PATH);
		// loadData();
		return confMatrix;
	}

	public void setConfMatrix(ConfusionMatrix confMatrix) {
		this.confMatrix = confMatrix;
	}

	/*
	 * Uses the train file to train the model. Then uses the trained model to
	 * compute predictions, confusion matrix on the holdout data
	 */
	public void loadData() {
		blnBoost = false;
		proxy = new ClassifierProxy();
		InputData in = null;
		try {
			System.out.println("calling intial state");
			in = proxy.computeInitialState(trainFile, holdoutFile);
			setConfMatrix(in.getConfMatrix());
			setParams(in.getParams());
			setPredInstances(in.getPredInstances());
			setDispMatrix(confMatrix);
			setBiasMatrix(dispMatrix);

			// proxy.validateTraining(in);
			setTrainConfMatrix(in.getTrainConfMatrix());
			setTrainPredInstances(in.getTrainPredInstances());
		} catch (Exception e) {
			e.printStackTrace();
			error = true;
			setErrorLog(e);
		}

		MetaChartBean chart = ((MetaChartBean) BeanFinder
				.findBean(MetaConstants.BEAN_DIVERGENCE_CHART));
		if (null != chart)
			chart.reset();
	}

	public boolean loadData(String inFile) {
		blnBoost = false;
		boolean blnStatus = true;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(inFile));
			setPredInstances(PredictionsLoader.load(reader));
			setConfMatrix(ConfusionMatrixLoader.load(reader));
			setParams(ModelParamLoader.load(reader));
			setDispMatrix(confMatrix);
			setBiasMatrix(dispMatrix);
		} catch (IOException e) {
			blnStatus = false;
		}
		return blnStatus;
	}

	public void setDispMatrix(ConfusionMatrix mat) {
		int[][] c = mat.getMatrix();
		int[][] b = new int[mat.getNumLables()][mat.getNumLables()];
		for (int i = 0; i < mat.getNumLables(); i++)
			for (int j = 0; j < mat.getNumLables(); j++)
				b[i][j] = c[i][j];
		dispMatrix = new DisplayMatrix();
		dispMatrix.setMatrix(b);
	}

	public void setBiasMatrix(DisplayMatrix mat) {
		int[][] c = mat.getMatrix();
		int[][] b = new int[2][2];
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				b[i][j] = c[i][j];
		biasMatrix = new BiasMatrix();
		biasMatrix.setMatrix(b);
		biasMatrix.setNumLabels(2);
	}

	public String serialize() throws IOException {
		Writer strWriter = new StringWriter();
		serialize(strWriter);
		return strWriter.toString();
	}

	private void serialize(Writer inWriter) throws IOException {
		BufferedWriter writer = new BufferedWriter(inWriter);
		writer.append(MetaConstants.IN_PRED_BEGIN);
		for (InputPredictionInstance instance : predInstances) {
			writer.newLine();
			writer.append(instance.getTrueLabel() + " "
					+ instance.getPredLabel() + " ");
			for (double y : instance.getYi())
				writer.append(y + " ");
		}
		writer.newLine();
		writer.append(MetaConstants.IN_PRED_END);
		writer.newLine();

		writer.append(MetaConstants.IN_CONFUSION_BEGIN);
		writer.newLine();
		int[][] matrix = confMatrix.getMatrix();
		for (int i = 0; i < confMatrix.getNumLables(); i++)
			for (int j = 0; j < confMatrix.getNumLables(); j++)
				writer.append(matrix[i][j] + " ");
		writer.newLine();
		writer.append(MetaConstants.IN_CONFUSION_END);
		writer.newLine();

		writer.append(MetaConstants.IN_MODEL_BEGIN);
		writer.newLine();
		writer.append(params.getParams().length + " ");
		for (double param : params.getParams())
			writer.append(param + " ");
		writer.newLine();
		writer.append(MetaConstants.IN_MODEL_END);
		writer.newLine();

		writer.append(MetaConstants.IN_PARAM_BEGIN);
		writer.newLine();
		writer.append(params.getWParams().length + " ");
		for (double param : params.getWParams())
			writer.append(param + " ");
		writer.newLine();
		writer.append(MetaConstants.IN_PARAM_END);

		writer.close();
	}

	public void serialize(String filepath) throws IOException {
		serialize(new FileWriter(filepath));
	}

	/*
	 * Implements hyperparameter optimization based on current confusion matrix
	 * and the bias input
	 */
	public void update() throws Exception {
		blnBoost = false;
		setBiasMatrix(dispMatrix);
		CurrentState cstate = CurrentState
				.createCurrentState(getPredInstances());

		TargetStateCalculator tstateCalc = new TargetStateCalculator(this,
				cstate);
		TargetState tstate = tstateCalc.calculate();

		Optimizer optimizer = new Optimizer(this, cstate, tstate, proxy);
		ModelParams params = null;
		try {
			params = optimizer.optimize2();
		} catch (Exception e) {
			e.printStackTrace();
			params = null;
		}
		if (null != params) {
			setDispMatrix(confMatrix);
			proxy.validateTraining(this);
			String filename = MetaConstants.MODEL_DOWNLOAD_PATH + "optim_"
					+ System.currentTimeMillis();
			new MetaModelGenerator().serializeModel(proxy.getClassifier(),
					filename + ".model", filename + ".params");
			error = !params.isOptim();
		}
		if (null == params || error)
			errorLog = "Could not optimize parameters for this configuration";
	}

	public void createDefaultBiasLowFP(int fpCount) {
		setDispMatrix(confMatrix);
		setBiasMatrix(dispMatrix);
		biasMatrix.setFp(fpCount);
	}

	public void createDefaultBiasLowFN(int fnCount) {
		setDispMatrix(confMatrix);
		setBiasMatrix(dispMatrix);
		biasMatrix.setFn(fnCount);
	}

	public void growMB() throws Exception {
		blnBoost = true;
		MartiBoost mboost = new MartiBoost(trainFile, holdoutFile, mbLevels);
		mboost.build();
		mbGraph = MBGraphvisGenerator.generateGraph(mboost);
	}

	public void createBias(int targetFP, int targetFN) {
		System.out.println("Received bias : " + targetFP + ", " + targetFN);
		setDispMatrix(confMatrix);
		setBiasMatrix(dispMatrix);
		if (targetFP < biasMatrix.getFp())
			biasMatrix.setFp(targetFP);
		if (targetFN < biasMatrix.getFn())
			biasMatrix.setFn(targetFN);
		System.out.println("Set bias : " + biasMatrix.getFp() + ", "
				+ biasMatrix.getFn());
	}

	public void copyFrom(InputData model) throws Exception {
		this.confMatrix = model.getConfMatrix();
		setDispMatrix(confMatrix);
		setTrainConfMatrix(model.getTrainConfMatrix());
		setPredInstances(model.getPredInstances());
		setTrainPredInstances(model.getTrainPredInstances());
		setParams(model.getParams());
	}

	public void createBias() {
		setDispMatrix(confMatrix);
		setBiasMatrix(dispMatrix);
		biasMatrix.setFp(confMatrix.getFp() - 1);
		biasMatrix.setFn(confMatrix.getFn() - 1);

	}
}
