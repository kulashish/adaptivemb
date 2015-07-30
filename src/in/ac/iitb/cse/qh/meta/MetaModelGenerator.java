package in.ac.iitb.cse.qh.meta;

import in.ac.iitb.cse.qh.classifiers.ModifiedLogistic;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;

import Jama.Matrix;

public class MetaModelGenerator {
	private double[] modelWeights;
	private double[][] featureWeightsPerModel;

	public MetaModelGenerator() {

	}

	public MetaModelGenerator(double[] modelWeights,
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

	public ModifiedLogistic generate() {
		double[] metaModelWeights = calculateMetamodelWeights();
		ModifiedLogistic mLogistic = new ModifiedLogistic();
		mLogistic.setWparameters(metaModelWeights);
		mLogistic.setNumberofAttributes(metaModelWeights.length);
		return mLogistic;
	}

	public void serializeModel(ModifiedLogistic model, String filePath)
			throws FileNotFoundException, IOException {
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(
				filePath));
		oos.writeObject(model);
		oos.flush();
		oos.close();
	}

	public void serializeModel(ModifiedLogistic model, String filePath,
			String modelParamsFile) throws FileNotFoundException, IOException {
		serializeModel(model, filePath);
		BufferedWriter writer = new BufferedWriter(new FileWriter(
				modelParamsFile));
		for (double w : model.getWparameters())
			writer.append(w + " ");
		writer.newLine();
		writer.close();
	}

	public static void main(String[] args) {
		MetaModelGenerator modelGen = new MetaModelGenerator();
		modelGen.setModelWeights(new double[] { 1, 2, 3 });
		modelGen.setFeatureWeightsPerModel(new double[][] { { 1, 1, 2, 1 },
				{ 2, 1, 1, 2 }, { 1, 1, 1, 3 } });
		double[] meta = modelGen.calculateMetamodelWeights();
		for (double w : meta)
			System.out.print(w + ", ");
		ModifiedLogistic model = modelGen.generate();
		try {
			modelGen.serializeModel(model, "meta.model");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
