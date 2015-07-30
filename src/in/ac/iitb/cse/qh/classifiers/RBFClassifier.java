package in.ac.iitb.cse.qh.classifiers;

import in.ac.iitb.cse.qh.data.ConfusionMatrix;
import in.ac.iitb.cse.qh.util.WekaUtil;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instance;
import weka.core.Instances;

public class RBFClassifier {

	private SMO classifier;

	public RBFClassifier() {
		classifier = new SMO();
		Kernel rbf = new RBFKernel();
		classifier.setKernel(rbf);
	}

	public static void main(String[] args) {
		String trainFile = args[0];
		String testFile = args[1];
		try {
			RBFClassifier rbf = new RBFClassifier();
			Instances trainInst = WekaUtil.getInstances(trainFile);
			// Instances holdInst = WekaUtil.getInstances(holdoutFile);
			Instances testInst = WekaUtil.getInstances(testFile);

			rbf.classifier.getKernel().buildKernel(trainInst);
			rbf.buildClassifier(trainInst);
			ConfusionMatrix conf = rbf.evaluateModel(testInst);
			conf.display();
			int pos = conf.getTp() + conf.getFn();
			int neg = conf.getTn() + conf.getFp();
			float accPos = conf.getTp() * 1.0f / pos;
			float accNeg = conf.getTn() * 1.0f / neg;
			float acc = (conf.getTp() + conf.getTn()) * 1.0f / (pos + neg);
			System.out.println("Acc+ :" + accPos + ", Acc- :" + accNeg
					+ ", Acc :" + acc);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private ConfusionMatrix evaluateModel(Instances instances) throws Exception {
		ConfusionMatrix confMatrix = new ConfusionMatrix();
		double dist[][] = new double[instances.numInstances()][instances
				.numClasses()];

		int countFN = 0;
		int countFP = 0;
		int countTN = 0;
		int countTP = 0;
		double pred = 0.0d;
		double actual = 0.0d;
		int i = -1;

		classifier.getKernel().buildKernel(instances);
		for (Instance instance : instances) {
			dist[++i] = classifier.distributionForInstance(instance);

			pred = dist[i][0] >= dist[i][1] ? 0 : 1;
			// pred = dist[i][1] > 0 ? 1 : 0;
			actual = instance.classValue();

			if (pred != actual) {
				if (pred == 1)
					countFP++;
				else
					countFN++;
			} else {
				if (pred == 1)
					countTP++;
				else
					countTN++;
			}
		}

		int mat[][] = { { countTN, countFP }, { countFN, countTP } };
		confMatrix.setMatrix(mat);
		return confMatrix;

	}

	private void buildClassifier(Instances trainInst) throws Exception {
		classifier.buildClassifier(trainInst);

	}

}
