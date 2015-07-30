package in.ac.iitb.cse.qh.test;

import in.ac.iitb.cse.qh.classifiers.ModifiedLogistic;
import in.ac.iitb.cse.qh.util.MetaConstants;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class LogisticTester {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		Instances trainInstances;
		DataSource source;
		source = new DataSource(MetaConstants.TRAIN_FILE_PATH);
		trainInstances = source.getDataSet();
		trainInstances.setClassIndex(trainInstances.numAttributes()-1);
		ModifiedLogistic ml = new ModifiedLogistic();
		double[] d = new double[trainInstances.numAttributes()];
		for(int i=0;i<d.length;i++)
			d[i]=1;
		ml.setHyperparameters(d);
		ml.buildClassifier(trainInstances);
		
		double dist[][] = new double[trainInstances.numInstances()][trainInstances.numClasses()];

		int countFN=0;
		int countFP=0;
		int countTN=0;
		int countTP=0;

		for(int i=0;i<trainInstances.numInstances();i++)
		{
			Instance instance = trainInstances.get(i);
			dist[i]=ml.distributionForInstance(instance);

			//double pred = dist[i][0]>=dist[i][1]?0:1;
			double pred = dist[i][1]>0?1:0;

			if(pred != instance.classValue())
			{
				if(pred==1)
					countFP++;
				else
					countFN++;
			}
			else
			{
				if(pred==1)
					countTP++;
				else
					countTN++;				
			}
			System.out.println(dist[i][0]+" "+dist[i][1]+" "+instance.classValue()+" " + pred);
			
		}
		System.out.println(countTN+" "+countFP+"\n"+countFN+" "+countTP);
		System.out.println("countFP="+countFP);
		System.out.println("%FP="+countFP*200.0/trainInstances.numInstances());
		System.out.println("countFN="+countFN);
		System.out.println("%FN="+countFN*200.0/trainInstances.numInstances());
	}

}
