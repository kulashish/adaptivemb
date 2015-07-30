package in.ac.iitb.cse.qh.util;

import in.ac.iitb.cse.qh.classifiers.ModifiedLogistic;
import in.ac.iitb.cse.qh.meta.MetaModelGenerator;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class MetaModelGeneratorHelper {

	public static double[][] readWeightsFromFile(String filePath, int nf, int nm)
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

	public static double[] readWeightsFromFile(String filePath, int n)
			throws IOException {
		return readWeightsFromFile(filePath, n + 1, 1)[0];
	}

	private static double[] parseWeights(String line) {
		String[] strWeights = line.split(" ");
		double[] w = new double[strWeights.length];
		int index = 0;
		for (String strWeight : strWeights)
			w[index++] = Double.parseDouble(strWeight);
		return w;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String featureWeightsPerModelFilePath = args[0];
		String modelWeightsFilePath = args[1];
		String modelFile = args[2];
		String metaModelParamsFile = args[3];
		// int numFeatures = Integer.parseInt(args[4]);
		int numModels = Integer.parseInt(args[4]);

		try {
			/*
			 * Removed numFeatures from the parameter list however retaining the
			 * method signature; hence passing numFeatures as -1.
			 */
			double[][] featureWeightsPerModel = readWeightsFromFile(
					featureWeightsPerModelFilePath, -1, numModels);
			double[] modelWeights = readWeightsFromFile(modelWeightsFilePath,
					numModels);
			MetaModelGenerator modelGen = new MetaModelGenerator(modelWeights,
					featureWeightsPerModel);
			ModifiedLogistic model = modelGen.generate();
			modelGen.serializeModel(model, modelFile, metaModelParamsFile);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

}
