package in.ac.iitb.cse.qh.data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import in.ac.iitb.cse.qh.meta.ClassifierProxy;
import in.ac.iitb.cse.qh.meta.Optimizer;
import in.ac.iitb.cse.qh.meta.TargetStateCalculator;
import in.ac.iitb.cse.qh.util.MetaConstants;

public class ModelData {

	/**
	 * @param args
	 */
	
	private int maxIterations = 10;
	private ModelParams[] modelParArr;
	private int curSize;
	
	private BiasMatrix biasMatrix;
	private DisplayMatrix dispMatrix;
	
	private double initialStepSize = MetaConstants.INITIAL_STEP;
	
	public ModelData(int i)
	{
		curSize=0;
		maxIterations=i;
		modelParArr = new ModelParams[maxIterations];
	}
	
//	public void insert(ModelParams m)
//	{
//		modelParArr[curSize]=m;
//		curSize++;
//	}
	
	public void insertAt(ModelParams m, int i)
	{
		modelParArr[i]=m;
		curSize=i+1;
	}
	
	public ModelParams getModelAt(int index)
	{
		return modelParArr[index];
	}
	
	public int getSize()
	{
		return curSize;
	}
	
	public double getInitialStepSize() {
		return initialStepSize;
	}

	public void setInitialStepSize(double initialStepSize) {
		System.out.println("setting intial step size: " + initialStepSize);
		this.initialStepSize = initialStepSize;
	}

	public int getMaxIterations() {
		return maxIterations;
	}

	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}

	public DisplayMatrix getDispMatrix() {
		if (null == dispMatrix)
			loadData(MetaConstants.IN_FILE_PATH);
		return dispMatrix;
	}

	public void setDispMatrix(DisplayMatrix dispMatrix) {
		this.dispMatrix = dispMatrix;
	}

//	public BiasMatrix getBiasMatrix() {
//		if (null == biasMatrix)
//			loadData(MetaConstants.IN_FILE_PATH);
//		return biasMatrix;
//	}

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

	public boolean loadData(String inFile) {
		boolean blnStatus = true;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(inFile));
//			setPredInstances(PredictionsLoader.load(reader));
//			setConfMatrix(ConfusionMatrixLoader.load(reader));
//			setParams(ModelParamLoader.load(reader));
//			setDispMatrix(confMatrix);
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

	public void serialize(String filepath) throws IOException {
	BufferedWriter writer = new BufferedWriter(new FileWriter(filepath));
	writer.append(MetaConstants.IN_PRED_BEGIN);
	ModelParams mp=modelParArr[curSize-1];
	List<InputPredictionInstance> predInstances = mp.getPredInstances();
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
	int[][] matrix = mp.getConfMatrix().getMatrix();
	for (int i = 0; i < mp.getConfMatrix().getNumLables(); i++)
		for (int j = 0; j < mp.getConfMatrix().getNumLables(); j++)
			writer.append(matrix[i][j] + " ");
	writer.newLine();
	writer.append(MetaConstants.IN_CONFUSION_END);
	writer.newLine();

	writer.append(MetaConstants.IN_MODEL_BEGIN);
	writer.newLine();
	writer.append(mp.getParams().length + " ");
	for (double param : mp.getParams())
		writer.append(param + " ");
	writer.newLine();
	writer.append(MetaConstants.IN_MODEL_END);

	writer.append(MetaConstants.IN_WMODEL_BEGIN);
	writer.newLine();
	writer.append(mp.getWParams().length + " ");
	for (double param : mp.getWParams())
		writer.append(param + " ");
	writer.newLine();
	writer.append(MetaConstants.IN_WMODEL_END);
	
	writer.close();
}

//public String update() {
//	String result = "success";
//	setBiasMatrix(dispMatrix);
//	ModelParams mp=modelParArr[curSize-1];
//	List<InputPredictionInstance> predInstances = mp.getPredInstances();
//	CurrentState cstate = CurrentState
//			.createCurrentState(predInstances);
//
//	TargetStateCalculator tstateCalc = new TargetStateCalculator(this,null,
//			cstate);
//	TargetState tstate = tstateCalc.calculate();
//
//	Optimizer optimizer = new Optimizer(null, cstate, tstate,
//			null, this);
//	ModelParams params = null;
//	try {
//		params = optimizer.optimize();
//	} catch (Exception e) {
//		params = null;
//	}
//	if (null != params)
//		setDispMatrix(confMatrix);
//	else
//		result = "failure";
//
//	return result;
//}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
