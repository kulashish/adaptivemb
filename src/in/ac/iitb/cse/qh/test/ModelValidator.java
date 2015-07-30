package in.ac.iitb.cse.qh.test;

import in.ac.iitb.cse.qh.data.InputData;
import in.ac.iitb.cse.qh.data.InputPredictionInstance;
import in.ac.iitb.cse.qh.meta.ClassifierProxy;
import in.ac.iitb.cse.qh.util.WekaUtil;

import java.io.File;
import java.io.IOException;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class ModelValidator {

	public static void main(String[] args) {
		InputData in = new InputData();
		// System.out.println("Loading Input data");
//		String modelFile = args[0];
//		String trainFile = args[1];
		String trainFile = "C:\\Users\\akulkarni\\Documents\\Personal\\PhD\\AAAI\\Dataset\\spam-train\\uci-train-11.arff";
//		String holdoutFile = args[2];
		String holdoutFile = "C:\\Users\\akulkarni\\Documents\\Personal\\PhD\\AAAI\\Dataset\\spam-holdout\\uci-holdout-11.arff";
//		in.loadData(args[0]); // load input data;
		ClassifierProxy proxy = new ClassifierProxy();
		try {
			Instances trainInstances = WekaUtil.getInstances(trainFile);
			proxy.setTrainingInstances(trainInstances); // Training file
			Instances holdoutInstances = WekaUtil.getInstances(holdoutFile);
			proxy.setHoldoutInstances(holdoutInstances);
			in = proxy.computeNewState(in.getParams());
//			proxy.validateTraining(in);
			Instances train0 = new Instances(trainInstances, 0);
			Instances train1 = new Instances(trainInstances, 0);
			split(in.getTrainPredInstances(), trainInstances, train0, train1);
			save(train0, "train0.arff");
			save(train1, "train1.arff");

//			Instances holdoutInstances = WekaUtil.getInstances(holdoutFile);
			Instances hold0 = new Instances(holdoutInstances, 0);
			Instances hold1 = new Instances(holdoutInstances, 1);
			split(in.getPredInstances(), holdoutInstances, hold0, hold1);
			save(hold0, "hold0.arff");
			save(hold1, "hold1.arff");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void split(List<InputPredictionInstance> labels,
			Instances instances, Instances zero, Instances one) {
		Instance instance = null;
		for (int i = 0; i < instances.numInstances(); i++) {
			instance = instances.get(i);
			if (labels.get(i).getPredLabel() == 0)
				zero.add(instance);
			else
				one.add(instance);
		}
	}

	private static void save(Instances instances, String file)
			throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		saver.setFile(new File(file));
		// saver.setDestination(new File("./data/test.arff")); // **not**
		// necessary in 3.5.4 and later
		saver.writeBatch();
	}
}
