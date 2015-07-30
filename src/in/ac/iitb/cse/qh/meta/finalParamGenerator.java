package in.ac.iitb.cse.qh.meta;

import in.ac.iitb.cse.qh.classifiers.ModifiedLogistic;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import Jama.Matrix;

public class finalParamGenerator {

	/**
	 * @param args
	 */
	private double[] modelWeights;
	private double[][] featureWeightsPerModel;
//	private double[] finalWeights;
	
	public finalParamGenerator() {
		
	}
	
	public finalParamGenerator(double[] modelWeights,
			double[][] featureWeightsPerModel) {
		this.modelWeights = modelWeights;
		this.featureWeightsPerModel = featureWeightsPerModel;
	}

	public void setModelWeights(double[] modelWeights) {
		this.modelWeights = modelWeights;
	}
	
	public void setFeatureWeightsPerModel(double[][] featureWeightsPerModel) {
		this.featureWeightsPerModel = featureWeightsPerModel;
	}

	private double[] calculateMetamodelWeights() {
		Matrix modelWeightsVector = new Matrix(modelWeights, 1);
		Matrix featureWeightsPerModelMatrix = new Matrix(featureWeightsPerModel);
		Matrix metaModelWeightsVector = modelWeightsVector
				.times(featureWeightsPerModelMatrix);
		return metaModelWeightsVector.getColumnPackedCopy();
	}

	private static double[] parseWeights(String line) {
		String[] strWeights = line.split(" ");
		double[] w = new double[strWeights.length];
		int index = 0;
		for (String strWeight : strWeights)
			w[index++] = Double.parseDouble(strWeight);
		return w;
	}

	public static double[][] readWeightsFromFile(String filePath, int nm)
			throws IOException {
		double[][] w = null;
		double[] w0 = null;
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		for (int row = 0; row < nm; row++) {
			if (row == 0) {
				w0 = parseWeights(reader.readLine());
				w = new double[nm][w0.length];
				w[row] = w0;
			} else
				w[row] = parseWeights(reader.readLine());
		}
		reader.close();
		return w;
	}

	public static double[] readWeightsFromFile(String filePath)
			throws IOException {
		
		double[] w0 = null;
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		w0 = parseWeights(reader.readLine());
		reader.close();
		return w0;
	}
	
	public static double[] finalModelGenetrator(double[] modelWeights, double[][] featureWeightsPerModel)
	{
		double[] finalWeights= new double[featureWeightsPerModel[0].length];
		finalWeights[0]=modelWeights[0];
		for(int j=1;j<modelWeights.length;j++)
		{
			finalWeights[0]+=modelWeights[j]*featureWeightsPerModel[j-1][0];
		}
		for(int i=1;i<finalWeights.length;i++)
		{
			finalWeights[i]=0;
			for(int j=0;j<featureWeightsPerModel.length;j++)
			{
				finalWeights[i]+=modelWeights[j+1]*featureWeightsPerModel[j][i];
			}
		}
		
		return finalWeights;
	}
	

	public static ModifiedLogistic generate(double[] metaModelWeights) {
		ModifiedLogistic mLogistic = new ModifiedLogistic();
		mLogistic.setWparameters(metaModelWeights);
		mLogistic.setNumberofAttributes(metaModelWeights.length);
		return mLogistic;
	}

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub

		String featureWeightsPerModelFilePath = args[0];
		String modelWeightsFilePath = args[1];
		String modelFile = args[2];
		String metaModelParamsFile = args[3];
		int numModels = Integer.parseInt(args[4]);

		double[][] featureWeightsPerModel = readWeightsFromFile(
				featureWeightsPerModelFilePath, numModels);
		double[] modelWeights = readWeightsFromFile(modelWeightsFilePath);
		double[] finalWeights = finalModelGenetrator(modelWeights, featureWeightsPerModel);
		ModifiedLogistic ml = generate(finalWeights);
		
		MetaModelGenerator modelGen = new MetaModelGenerator(modelWeights,
				featureWeightsPerModel);
		modelGen.serializeModel(ml, modelFile, metaModelParamsFile);

	}

}
