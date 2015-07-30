package in.ac.iitb.cse.qh.naiveBayes;

import in.ac.iitb.cse.qh.data.ConfusionMatrix;
import in.ac.iitb.cse.qh.data.InputData;
import in.ac.iitb.cse.qh.data.InputPredictionInstance;
import in.ac.iitb.cse.qh.data.ModelParams;
import in.ac.iitb.cse.qh.util.MetaConstants;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class NaiveBayesParam {

	/**
	 * @param args
	 */
	private String trainFile;
	private String testFile;

	private Instances trainInstances;
	private Instances testInstances;
 
	private double[][] mean;
	private double[][] variance;
	private double[] classProb;// Note classProb is now only class frequency
	private double alpha;
	
	private double[] param;
	
	public NaiveBayesParam(String trainFile, String testFile) {
		this.trainFile = trainFile;
		this.testFile = testFile;

		try {
			DataSource source;
			source = new DataSource(trainFile);
			trainInstances = source.getDataSet();
			source = new DataSource(testFile);
			testInstances = source.getDataSet();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		// if (train.classIndex() == -1)
		trainInstances.setClassIndex(trainInstances.numAttributes() - 1);
		testInstances.setClassIndex(testInstances.numAttributes() - 1);
		mean = new double[trainInstances.numAttributes()-1][trainInstances
				.numClasses()];
		variance = new double[trainInstances.numAttributes()-1][trainInstances
				.numClasses()];
		classProb = new double[trainInstances.numClasses()];

		//param= new double[2*(trainInstances.numAttributes()-1)*trainInstances.numClasses()+trainInstances.numClasses()];
		param = new double[1];
		alpha =10d;
		param[0]=10d;
		calculateMeanVar();
		//moveParToArray();
	}

	private void calculateMeanVar() {
		for (int i = 0; i < trainInstances.numInstances(); i++) {
			Instance instance = trainInstances.get(i);
			for (int j = 0; j < trainInstances.numAttributes() - 1; j++) {
				mean[j][(int) instance.classValue()] += instance.value(j);
				classProb[(int) instance.classValue()]++;
				variance[j][(int) instance.classValue()] += instance.value(j)
						* instance.value(j);
			}
		}
		for (int j = 0; j < trainInstances.numAttributes() - 1; j++) {
			for (int i = 0; i < trainInstances.numClasses(); i++) {
				mean[j][i] = mean[j][i] / classProb[i];
				variance[j][i] = variance[j][i] / classProb[i] - mean[j][i]
						* mean[j][i];
				
				//System.out.println("mean["+j+"]["+i+"]="+mean[j][i]+",variance["+j+"]["+i+"]="+variance[j][i]);
			}
		}
		//for (int i = 0; i < trainInstances.numClasses(); i++) {
		//	classProb[i] /= trainInstances.numInstances();
		//}
	}

	private double prob(double x, double mean, double variance) {
		if(variance < 0.0001d)
		{
			if(x > mean - 0.0001d && x < mean +0.0001d)
				return 1d;
			else return 0d;
		}
		return Math.pow(Math.E, -(x - mean) * (x - mean) / (2 * variance))
				/ Math.sqrt(2 * Math.PI * variance);
	}
/*
	private double problogderWRTmean(double x, double mean, double variance) {
		if(variance < 0.0001d)
			return 0d;
		return  (x-mean)/variance;
	}
	
	private double problogderWRTvar(double x, double mean, double variance) {
		if(variance < 0.0001d)
			return 0d;
		double sd=Math.sqrt(variance);
		return  -1/sd+(x - mean) * (x - mean)/(variance*sd);
	}
*/
	private double problog(double x, double mean, double variance) {
		if(variance < 0.0001d)
			return 0d;
		return  -(x - mean) * (x - mean) / (2 * variance)
				- Math.log( Math.sqrt(2 * Math.PI * variance));
	}
	
	public double[][] calcJacobian(Instance instance) {
		int height=trainInstances.numClasses();
		double jac[][]= new double[height][1];
		
		jac[0][0]=1.0/(classProb[0]+alpha)-2/(classProb[0]+classProb[1]+2*alpha);
		jac[1][0]=1.0/(classProb[1]+alpha)-2/(classProb[0]+classProb[1]+2*alpha);
		return jac;
	}
	
	public double[][] calcJacobian(int index, double[] param)
	{
		int height=trainInstances.numClasses();
		double jac[][]= new double[height][1];
		
		jac[0][0]=1.0/(classProb[0]+param[0])-2/(classProb[0]+classProb[1]+2*param[0]);
		jac[1][0]=1.0/(classProb[1]+param[0])-2/(classProb[0]+classProb[1]+2*param[0]);
		return jac;
		
	}

	public double[][] calcJacobian(int index) {

		Instance instance = trainInstances.get(index);
		
		int height=trainInstances.numClasses();
		double jac[][]= new double[height][1];
		
		jac[0][0]=1.0/(classProb[0]+alpha)-2/(classProb[0]+classProb[1]+2*alpha);
		jac[1][0]=1.0/(classProb[1]+alpha)-2/(classProb[0]+classProb[1]+2*alpha);
		return jac;
	}
	/*	
	public double[][] calcJacobian(Instance instance) {
		
		int height=trainInstances.numClasses();
		int width=(trainInstances.numAttributes()-1)*4+trainInstances.numClasses();
		double jac[][]= new double[height][width];

		for(int i=0;i<height;i++)
		{
			int j=0;
			for(;j<width;j++)
			{
				if(i==0)
				{
					if(j<trainInstances.numAttributes()-1)
						jac[i][j]=problogderWRTmean(instance.value(j), mean[j][i], variance[j][i]);
					else if(j<2*trainInstances.numAttributes()-2)
						jac[i][j]=problogderWRTvar(instance.value(j-trainInstances.numAttributes()+1), mean[j-trainInstances.numAttributes()+1][i], variance[j-trainInstances.numAttributes()+1][i]);
					else if(j==2*trainInstances.numAttributes()-2)
						jac[i][j]=1/classProb[i];
					else
						jac[i][j]=0;
				}
				else
				{
					int gap=2*trainInstances.numAttributes()-1;
					if(j<gap)
						jac[i][j]=0;
					else if(j<gap+trainInstances.numAttributes()-1)
						jac[i][j]=problogderWRTmean(instance.value(j-gap), mean[j-gap][i], variance[j-gap][i]);
					else if(j < gap+2*trainInstances.numAttributes()-2)
						jac[i][j]=problogderWRTvar(instance.value(j-gap-trainInstances.numAttributes()+1), mean[j-gap-trainInstances.numAttributes()+1][i], variance[j-gap-trainInstances.numAttributes()+1][i]);
					else
						jac[i][j]=1/classProb[i];;
				}
			}
		}
		return jac;
	}
*/	

/*	
	public double[][] calcJacobian(int index, double[] param)
	{
		Instance instance = trainInstances.get(index);
		int height=trainInstances.numClasses();
		int width=(trainInstances.numAttributes()-1)*4+trainInstances.numClasses();
		double jac[][]= new double[height][width];

		for(int i=0;i<height;i++)
		{
			int j=0;
			for(;j<width;j++)
			{
				if(i==0)
				{
					if(j<trainInstances.numAttributes()-1)
						jac[i][j]=problogderWRTmean(instance.value(j), param[j+i*(2*trainInstances.numAttributes()-1)], param[j+trainInstances.numAttributes()-1+i*(2*trainInstances.numAttributes()-1)]);
					else if(j<2*trainInstances.numAttributes()-2)
						jac[i][j]=problogderWRTvar(instance.value(j-trainInstances.numAttributes()+1), param[j-trainInstances.numAttributes()+1+i*(2*trainInstances.numAttributes()-1)], param[j+i*(2*trainInstances.numAttributes()-1)]);
					else if(j==2*trainInstances.numAttributes()-2)
						jac[i][j]=1/param[j];
					else
						jac[i][j]=0;
				}
				else
				{
					int gap=2*trainInstances.numAttributes()-1;
					if(j<gap)
						jac[i][j]=0;
					else if(j<gap+trainInstances.numAttributes()-1)
//						jac[i][j]=problogderWRTmean(instance.value(j-gap), param[j-gap+i*(2*trainInstances.numAttributes()-1)], param[j-gap+trainInstances.numAttributes()-1+i*(2*trainInstances.numAttributes()-1)]);
						jac[i][j]=problogderWRTmean(instance.value(j-gap), param[j], param[j+trainInstances.numAttributes()-1]);				
					else if(j < gap+2*trainInstances.numAttributes()-2)
//						jac[i][j]=problogderWRTvar(instance.value(j-gap-trainInstances.numAttributes()+1), param[j-gap-trainInstances.numAttributes()+1+i*(2*trainInstances.numAttributes()-1)], param[j-gap+i*(2*trainInstances.numAttributes()-1)]);
						jac[i][j]=problogderWRTvar(instance.value(j-gap-trainInstances.numAttributes()+1), param[j-trainInstances.numAttributes()+1], param[j]);
					else
						jac[i][j]=1/param[j];
				}
			}
		}
		
		return jac;
	}
*/	
	/*
	public double[][] calcJacobian(int index) {

		Instance instance = trainInstances.get(index);
		
		int height=trainInstances.numClasses();
		int width=(trainInstances.numAttributes()-1)*4+trainInstances.numClasses();
		double jac[][]= new double[height][width];

		for(int i=0;i<height;i++)
		{
			int j=0;
			for(;j<width;j++)
			{
				if(i==0)
				{
					if(j<trainInstances.numAttributes()-1)
						jac[i][j]=problogderWRTmean(instance.value(j), mean[j][i], variance[j][i]);
					else if(j<2*trainInstances.numAttributes()-2)
						jac[i][j]=problogderWRTvar(instance.value(j-trainInstances.numAttributes()+1), mean[j-trainInstances.numAttributes()+1][i], variance[j-trainInstances.numAttributes()+1][i]);
					else if(j==2*trainInstances.numAttributes()-2)
						jac[i][j]=1/classProb[i];
					else
						jac[i][j]=0;
				}
				else
				{
					int gap=2*trainInstances.numAttributes()-1;
					if(j<gap)
						jac[i][j]=0;
					else if(j<gap+trainInstances.numAttributes()-1)
						jac[i][j]=problogderWRTmean(instance.value(j-gap), mean[j-gap][i], variance[j-gap][i]);
					else if(j < gap+2*trainInstances.numAttributes()-2)
						jac[i][j]=problogderWRTvar(instance.value(j-gap-trainInstances.numAttributes()+1), mean[j-gap-trainInstances.numAttributes()+1][i], variance[j-gap-trainInstances.numAttributes()+1][i]);
					else
						jac[i][j]=1/classProb[i];
				}
			}
		}
		return jac;
	}
	*/
	/*
	private double[] moveParToArray()
	{
		//double param[] =new double[2*trainInstances.numClasses()*trainInstances.numAttributes()+trainInstances.numClasses()];
		for(int i=0;i<trainInstances.numClasses();i++)
		{
			int j=0;
			for(;j<trainInstances.numAttributes()-1;j++)
			{
				param[j+i*(2*trainInstances.numAttributes()-1)]=mean[j][i];
			}
			for(;j<2*trainInstances.numAttributes()-2;j++)
			{
				param[j+i*(2*trainInstances.numAttributes()-1)]=variance[j-trainInstances.numAttributes()+1][i];
			}
			param[j+i*(2*trainInstances.numAttributes()-1)]=classProb[i];
		}
		return param;
	}
	*/
	public InputData calcInitialState()
	{
		//double[] param=moveParToArray();
		
		//return calcNewState(param);
		return calcNewState(param);
	}
	public InputData calcNewState(double[] param)
	{
		InputData dat = new InputData();
		List<InputPredictionInstance> predInst = new ArrayList<InputPredictionInstance>();
		ConfusionMatrix confMatrix= new ConfusionMatrix();
		ModelParams modPar = new ModelParams();
		modPar.setParams(param);
		
		//updateParam(param);
		alpha = param[0];
		try {
			double dist[][] = distributionForInstances(trainInstances);
			int countFN=0;
			int countFP=0;
			int countTN=0;
			int countTP=0;
			System.out.println("1.# prob0 - prob1 - actual - pred");
			for (int i = 0; i < trainInstances.numInstances(); i++)
			{
				double pred;
				double actual = trainInstances.instance(i).classValue();
				//System.out.print("actual="+actual);
				if(dist[i][0] >= dist[i][1])
					pred=0d;
				else
					pred=1d;
		        //if (pred != actual)
		        if (!(pred > actual-0.1d && pred < actual + 0.1d))
		        {
//		        	if(pred == 1)
		        	if(pred > 0.9d && pred < 1.1d)
		        		countFP++;
		        	else
		        		countFN++;
					/*System.out.print(dist[i][0]);
		            System.out.print(" - ");
					System.out.print(dist[i][1]);
		            System.out.print(" - ");
		            System.out.print(actual);
		            System.out.print(" - ");
		            System.out.print(pred);
		            System.out.println(" - ");*/
		        }
		        else
		        {
		        	if(actual == 1.0)
		        		countTP++;
		        	else
		        		countTN++;
		        }
		        InputPredictionInstance inPredIns = new InputPredictionInstance((int)actual,(int) pred, dist[i]);
		        predInst.add(inPredIns);
			}
			//System.out.println("countFP="+countFP);
			//System.out.println("%FP="+countFP*200.0/trainInstances.numInstances());
			//System.out.println("countFN="+countFN);
			//System.out.println("%FN="+countFN*200.0/trainInstances.numInstances());
	
			int mat[][]={{countTN,countFP},{countFN,countTP}};
			confMatrix.setMatrix(mat);
			dat.setConfMatrix(confMatrix);
			dat.setParams(modPar);
			dat.setPredInstances(predInst);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return dat;
	}
/*	
	public InputData calcNewState(double[] param)
	{
		InputData dat = new InputData();
		List<InputPredictionInstance> predInst = new ArrayList<InputPredictionInstance>();
		ConfusionMatrix confMatrix= new ConfusionMatrix();
		ModelParams modPar = new ModelParams();
		modPar.setParams(param);
		
		updateParam(param);
		try {
			double dist[][] = distributionForInstances(trainInstances);
			int countFN=0;
			int countFP=0;
			int countTN=0;
			int countTP=0;
			System.out.println("1.# prob0 - prob1 - actual - pred");
			for (int i = 0; i < trainInstances.numInstances(); i++)
			{
				double pred;
				double actual = trainInstances.instance(i).classValue();
				//System.out.print("actual="+actual);
				if(dist[i][0] >= dist[i][1])
					pred=0d;
				else
					pred=1d;
		        //if (pred != actual)
		        if (!(pred > actual-0.1d && pred < actual + 0.1d))
		        {
//		        	if(pred == 1)
		        	if(pred > 0.9d && pred < 1.1d)
		        		countFP++;
		        	else
		        		countFN++;
					System.out.print(dist[i][0]);
		            System.out.print(" - ");
					System.out.print(dist[i][1]);
		            System.out.print(" - ");
		            System.out.print(actual);
		            System.out.print(" - ");
		            System.out.print(pred);
		            System.out.println(" - ");
		        }
		        else
		        {
		        	if(actual == 1.0)
		        		countTP++;
		        	else
		        		countTN++;
		        }
		        InputPredictionInstance inPredIns = new InputPredictionInstance((int)actual,(int) pred, dist[i]);
		        predInst.add(inPredIns);
			}
			//System.out.println("countFP="+countFP);
			//System.out.println("%FP="+countFP*200.0/trainInstances.numInstances());
			//System.out.println("countFN="+countFN);
			//System.out.println("%FN="+countFN*200.0/trainInstances.numInstances());
	
			int mat[][]={{countTN,countFP},{countFN,countTP}};
			confMatrix.setMatrix(mat);
			dat.setConfMatrix(confMatrix);
			dat.setParams(modPar);
			dat.setPredInstances(predInst);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return dat;
	}
*/
	/*
	private void updateParam(double[] param)
	{
		for(int i=0;i<trainInstances.numClasses();i++)
		{
			int j=0;
			for(;j<trainInstances.numAttributes()-1;j++)
			{
				mean[j][i]=param[j+i*(2*trainInstances.numAttributes()-1)];
			}
			for(;j<2*trainInstances.numAttributes()-2;j++)
			{
				variance[j-trainInstances.numAttributes()+1][i]=param[j+i*(2*trainInstances.numAttributes()-1)];
			}
			classProb[i]=param[j+i*(2*trainInstances.numAttributes()-1)];
		}
	}
	*/
	private double[][] distributionForInstances(Instances instances)
			throws Exception {

		double dist[][] = new double[instances.numInstances()][instances.numClasses()];

		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < instances.numClasses(); j++) {
				dist[i][j] = Math.log(classProb[j]+alpha/(classProb[0]+classProb[1]+2*alpha));
				/*if (classProb[j] > 0.0001d)
					dist[i][j] = Math.log(classProb[j]);
				else
					dist[i][j] = Math.log(0.0001d);*/
					//dist[i][j] = 0;
				//dist[i][j] = classProb[j];
			}
			Instance instance = instances.get(i);
			Enumeration enumAtts = instance.enumerateAttributes();
			int attIndex = 0;
			while (enumAtts.hasMoreElements()) {
				Attribute attribute = (Attribute) enumAtts.nextElement();
				if (!instance.isMissing(attribute)) {
					double temp, max = 0d;
					for (int j = 0; j < instances.numClasses(); j++) {
						temp = Math.max(
								1e-75,
								prob(instance.value(attribute),
										mean[attIndex][j],
										variance[attIndex][j]));
						//dist[i][j] *= temp;
						temp = problog(instance.value(attribute),
										mean[attIndex][j],
										variance[attIndex][j]);
						dist[i][j] += temp;
						if (dist[i][j] > max) {
							max = dist[i][j];
						}
					}
					/*if ((max > 0) && (max < 1e-75)) { // Danger of probability
														// underflow
						for (int j = 0; j < instance.numClasses(); j++) {
							dist[i][j] *= 1e75;
						}
					}*/
				}
				attIndex++;
				//System.out.println("attIndex="+attIndex);
			}
			//Utils.normalize(dist[i]);
		}

		return dist;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		long startTime = System.currentTimeMillis();

		//NaiveBayesParam nb = new NaiveBayesParam("data/split1.arff","data/split1_test.arff");
		//NaiveBayesParam nb = new NaiveBayesParam("/media/F/Acads/iitb/mtp/naivebayesdataset/bronchiolitis/br-processed.arff","" +
		//		"/media/F/Acads/iitb/mtp/naivebayesdataset/bronchiolitis/br-processed.arff");
		//NaiveBayesParam nb = new NaiveBayesParam("/media/F/Acads/iitb/mtp/naivebayesdataset/01/sylva_agnostic_valid.arff","" +
		//		"/media/F/Acads/iitb/mtp/naivebayesdataset/01/sylva_agnostic_valid.arff");

		// NaiveBayesParam nb = new NaiveBayesParam("data/split1lac.arff",
		// "data/split1lacTest.arff");
		//NaiveBayesParam nb = new NaiveBayesParam("data/split1b.arff", "data/split1bTest.arff");
		// NaiveBayesParam nb = new NaiveBayesPFaram("data/split1lac.arff",
		// "data/split1bTest.arff");
		//NaiveBayesParam nb = new NaiveBayesParam("data/split1lac.arff","data/split1hTest.arff");
		
		
		NaiveBayesParam nb = new NaiveBayesParam("/media/F/Acads/iitb/mtp/QuickHeal/data/split1per1.arff", "/media/F/Acads/iitb/mtp/QuickHeal/data/split1.arff");
				
		InputData initialData = nb.calcInitialState();
		try {
			initialData.serialize("/home/agam/nb.dat");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		/*nb.calcNewState(nb.param);


		try {
			double dist1[][] = nb.distributionForInstances(nb.trainInstances);
			double dist2[][] = nb.distributionForInstances(nb.testInstances);
			
			int countFN=0;
			int countFP=0;
			System.out.println("1.# prob0 - prob1 - actual - pred");
			for (int i = 0; i < nb.trainInstances.numInstances(); i++)
			{
				double pred;
				double actual = nb.trainInstances.instance(i).classValue();
				//System.out.print("actual="+actual);
				if(dist1[i][0] >= dist1[i][1])
					pred=0d;
				else
					pred=1d;
		        //if (pred != actual)
		        if (!(pred > actual-0.1d && pred < actual + 0.1d))
		        {
//		        	if(pred == 1)
		        	if(pred > 0.9d && pred < 1.1d)
		        		countFP++;
		        	else
		        		countFN++;
		        }
			}
			System.out.println("countFP="+countFP);
			System.out.println("%FP="+countFP*200.0/nb.trainInstances.numInstances());
			System.out.println("countFN="+countFN);
			System.out.println("%FN="+countFN*200.0/nb.trainInstances.numInstances());
			
			countFN=0;
			countFP=0;
			System.out.println("2.# prob0 - prob1 - actual - pred");
			for (int i = 0; i < nb.testInstances.numInstances(); i++)
			{
				double pred;
				double actual = nb.testInstances.instance(i).classValue();
				if(dist2[i][0] >= dist2[i][1])
					pred=0d;
				else
					pred=1d;
//		        if (pred != actual)
		        if (!(pred > actual-0.1d && pred < actual + 0.1d))
		        {
		        	if(pred > 0.9d && pred < 1.1d)
//		        	if(pred == 1)
		        		countFP++;
		        	else
		        		countFN++;
		        }
		        System.out.println("2.#"+dist2[i][0]+" "+dist2[i][1]+" " +actual+" "+pred);
			}
			System.out.println("countFP="+countFP);
			System.out.println("%FP="+countFP*100.0/nb.testInstances.numInstances());
			System.out.println("countFN="+countFN);
			System.out.println("%FN="+countFN*100.0/nb.testInstances.numInstances());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
*/
/*		
		double jac[][]=nb.calcJacobian(nb.trainInstances.instance(0));

		for(int i=0;i<nb.trainInstances.numClasses();i++)
		{
			for(int j=0;j<4*nb.trainInstances.numAttributes()-4+nb.trainInstances.numClasses();j++)
			{
				System.out.println("jac["+i+"]=["+j+"]="+jac[i][j]);
			}
		}
*/		
		long endTime = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		System.out.println(totalTime + " milliseconds");
	}
}