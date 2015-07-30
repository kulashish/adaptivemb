package in.ac.iitb.cse.qh.classifiers;

import in.ac.iitb.cse.qh.util.WekaUtil;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;

import weka.core.Instance;
import weka.core.Instances;

public class ClassifyDataFromModel {

	/**
	 * @param args
	 */

	ModifiedLogistic modLog;
	String modelFile;
	String testFile;
	Instances instances;
	double[] params;

	public ClassifyDataFromModel(String modelFile, String testFile)
			throws Exception {
		
		this.modelFile = modelFile;
		this.testFile = testFile;

		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(
				modelFile));
		modLog = (ModifiedLogistic) ois.readObject();
		ois.close();
		
		params=modLog.getWparameters();
		instances = WekaUtil.getInstances(testFile);
	}

	public void classifyAll() throws Exception {
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.get(i);
			if (modLog != null) {
				double[] dist = modLog.distributionForInstance(inst);
				for (int j = 0; j < dist.length; j++)
					System.out.println(dist[j]);
			} else {
				System.out.println("model null");
			}
			// System.out.println(modLog.classifyInstance(inst));
		}
	}

	public void classifyAllParams()
	{
		double sum=0;
		int counttp=0;
		int counttn=0;
		int countfp=0;
		int countfn=0;
		
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.get(i);
			sum=params[0];
//			for(int j=0;j<inst.numAttributes()-1;j++)
//				sum+=inst.value(j)*params[j+1];

			for(int j=1;j<params.length;j++)
				sum+=inst.value(j-1)*params[j];

			if(sum > 0)
			{
				System.out.println("predClass=yes score="+sum);
				if(inst.classValue() == 0)
				{
					countfp++;
				}
				else
					counttp++;
			}
			else
			{
				System.out.println("predClass=no score="+sum);
				if(inst.classValue() == 1)
				{
					countfn++;
				}
				else
					counttn++;
			}
		}
//		System.out.println("countyes="+countyes+" ,countno="+countno);
//		System.out.println("countfp="+countfp+" ,countfn="+countfn);
		
		System.out.println(counttn+" "+countfp);
		System.out.println(countfn+" "+counttp);
	}
	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		ClassifyDataFromModel classify = new ClassifyDataFromModel(args[0],
				args[1]);
		classify.classifyAllParams();
	}

}
