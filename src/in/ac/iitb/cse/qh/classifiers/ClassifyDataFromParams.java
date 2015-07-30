package in.ac.iitb.cse.qh.classifiers;

import in.ac.iitb.cse.qh.util.WekaUtil;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassifyDataFromParams {

	String paramFile;
	String testFile;
	Instances instances;
	String[] params;
	
	public ClassifyDataFromParams(String paramFile, String testFile) throws Exception
	{
		this.paramFile=paramFile;
		this.testFile=testFile;
		
//		DataSource datasource = new DataSource(testFile);
//		instances = datasource.getDataSet();

		instances = WekaUtil.getInstances(testFile);
		
		BufferedReader br = new BufferedReader(new FileReader(paramFile));
		params=br.readLine().split(" ");
		br.close();
	}
	
	public void classifyAll()
	{
		double sum=0;
		int counttp=0;
		int counttn=0;
		int countfp=0;
		int countfn=0;

		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.get(i);
			
			sum=Double.parseDouble(params[0]);
//			for(int j=0;j<inst.numAttributes();j++)
//				sum+=inst.value(j)*Double.parseDouble(params[j+1]);
			
			for(int j=1;j<params.length;j++)
				sum+=inst.value(j-1)*Double.parseDouble(params[j]);
			
			if(sum > 0)
			{
//				System.out.println("actual class="+inst.classValue()+ " predClass=0 score="+sum);
				if(inst.classValue() == 0)
				{
					counttn++;
				}
				else
					countfn++;
			}
			else
			{
//				System.out.println("actual class="+inst.classValue()+ " predClass=1 score="+sum);
				if(inst.classValue() == 1)
				{
					counttp++;
				}
				else
					countfp++;
			}
		}
//		System.out.println("countyes="+countyes+" ,countno="+countno);
//		System.out.println("countfp="+countfp+" ,countfn="+countfn);
		
		System.out.println(counttn+" "+countfp);
		System.out.println(countfn+" "+counttp);
	}
	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		ClassifyDataFromParams classify = new ClassifyDataFromParams(args[0],
				args[1]);
		classify.classifyAll();
	}

}