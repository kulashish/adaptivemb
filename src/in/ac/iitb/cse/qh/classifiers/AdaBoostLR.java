package in.ac.iitb.cse.qh.classifiers;

import in.ac.iitb.cse.qh.data.ConfusionMatrix;
import in.ac.iitb.cse.qh.util.WekaUtil;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instance;
import weka.core.Instances;

public class AdaBoostLR {

	private AdaBoostM1 classifier;

	public AdaBoostLR() {
		classifier = new AdaBoostM1();
		classifier.setClassifier(new ModifiedLogistic());
		classifier.setNumIterations(50);
	}

	public ConfusionMatrix evaluateModel(Instances instances) throws Exception {
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

	public static void main(String[] args) {
		String trainFile = args[0];
		String holdoutFile = args[1];
		String testFile = args[2];

		try {
			AdaBoostLR ada = new AdaBoostLR();
			Instances trainInst = WekaUtil.getInstances(trainFile);
			// Instances holdInst = WekaUtil.getInstances(holdoutFile);
			Instances testInst = WekaUtil.getInstances(testFile);

			ada.buildClassifier(trainInst);
			ConfusionMatrix conf = ada.evaluateModel(testInst);
			conf.display();
			int pos = conf.getTp()+conf.getFn();
			int neg = conf.getTn()+conf.getFp();
			float accPos = conf.getTp()*1.0f/pos;
			float accNeg = conf.getTn()*1.0f/neg;
			float acc = (conf.getTp()+conf.getTn())*1.0f/(pos+neg);
			System.out.println("Acc+ :"+accPos+", Acc- :"+accNeg+", Acc :"+acc);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	private void buildClassifier(Instances trainInst) throws Exception {
		classifier.buildClassifier(trainInst);
	}

}
