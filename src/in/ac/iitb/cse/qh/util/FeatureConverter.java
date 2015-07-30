package in.ac.iitb.cse.qh.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer.Context;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class FeatureConverter {

	// given a file of models
	// and an arff file
	// output-a new arff file with fn values as features

	String modelFile;
	String inputFile;
	String outputFile;
	String[] modelArray;
	List<String> modelList = new ArrayList<String>();

	public FeatureConverter(String modelFile, String inputFile,
			String outputFile) {
		this.modelFile = modelFile;
		this.inputFile = inputFile;
		this.outputFile = outputFile;
	}
	
	public FeatureConverter(String[] modelArray, String inputFile,
			String outputFile) {
		this.modelArray = modelArray;
		this.inputFile = inputFile;
		this.outputFile = outputFile;
	}

	public void convert() throws Exception {
		DataSource dt = null;
		Instances data = null;
		// try {
		dt = new DataSource(inputFile);
		data = dt.getDataSet();
		// } catch (Exception e) {
		// }
		FileReader fr = new FileReader(modelFile);
		BufferedReader br = new BufferedReader(fr);

		String line;

		Instances newData = new Instances(data, data.numInstances());

		System.out.print("newData.numAttributes()="+newData.numAttributes());
		
		while ((line = br.readLine()) != null) {
			modelList.add(line);
		}
		br.close();

		for (int i = 0; i < data.numInstances(); i++) {
			Instance inst = data.get(i);
			Instance newInstance = new DenseInstance(modelList.size());
			for (int j = 0; j < modelList.size(); j++) {
				String model = modelList.get(j);
				String[] params = model.split(" ");
				//System.out.println("\nparams.length="+params.length);
				//System.out.println("\ndata.numAttributes()="+data.numAttributes());
				double value = Double.parseDouble(params[0]);
//				if (params.length == data.numAttributes())
//					System.out.println("\nCorrect num of attributes");
				for (int k = 1; k < data.numAttributes(); k++) {
					value += inst.value(k-1) * Double.parseDouble(params[k]);
				}
				newInstance.insertAttributeAt(j + 1);
			}
			newData.add(newInstance);
		}

		// write instances to file
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newData);
		saver.setFile(new File(outputFile));
		saver.setDestination(new File(outputFile)); // **not** necessary in
													// 3.5.4 and later
		saver.writeBatch();
	}

	public void writefromFile() throws Exception {
		DataSource dt = null;
		Instances data = null;
		// try {
		dt = new DataSource(inputFile);
		data = dt.getDataSet();
		// } catch (Exception e) {
		// }
		FileReader fr = new FileReader(modelFile);
		BufferedReader br = new BufferedReader(fr);

		String line;

		FileWriter fw = new FileWriter(outputFile);
		BufferedWriter bw = new BufferedWriter(fw);
		
		bw.write(data.relationName()+"2\n\n");
		int att=1;
		while ((line = br.readLine()) != null) {
			modelList.add(line);
			bw.write("@attribute name"+att+" numeric\n");
			att++;
		}
		data.setClassIndex(data.numAttributes()-1);
		Attribute classAtt = data.classAttribute();
		bw.write("@attribute class {"+classAtt.value(0)+","+classAtt.value(1)+"}\n");
		bw.write("\n@data\n\n");
		br.close();
		
		for (int i = 0; i < data.numInstances(); i++) {
			Instance inst = data.get(i);
			for (int j = 0; j < modelList.size(); j++) {
				String model = modelList.get(j);
				String[] params = model.split(" ");
				//System.out.println("\nparams.length="+params.length);
				//System.out.println("\ndata.numAttributes()="+data.numAttributes());
				double value = Double.parseDouble(params[0]);
//				if (params.length == data.numAttributes())
//					System.out.println("\nCorrect num of attributes");
				for (int k = 1; k < data.numAttributes(); k++) {
					value += inst.value(k-1) * Double.parseDouble(params[k]);
				}
				bw.write(value+" ");
			}
			
			classAtt = inst.attribute(inst.numAttributes()-1);
			bw.write(classAtt.value((int)inst.value(inst.numAttributes()-1))+"\n");
		}

		bw.close();
	}


	public String writefromFiletoString(Context context) throws Exception {
		
		context.write("params", new Text("e1"));
		String output="";
		
		DataSource dt = null;
		Instances data = null;
		// try {
		dt = new DataSource(inputFile);
		context.write("params", new Text("e2"));
		data = dt.getDataSet();
		// } catch (Exception e) {
		// }
		
		if(data == null)
			context.write("params", new Text("data null"));
		else
			context.write("params", new Text("data not null"));
		FileReader fr=null;
		try{
			fr = new FileReader(modelArray[0]);	
		}
		catch(Exception e){
			context.write("params", new Text("Exception-"+e.toString()));
		}
		context.write("params", new Text("a0"));
		BufferedReader br = new BufferedReader(fr);

		context.write("params", new Text("a1"));
		
		String line;
		output=output+data.relationName()+"2\n\n";
		int att=1;
		while ((line = br.readLine()) != null) {
			modelList.add(line);
			output=output+"@attribute name"+att+" numeric\n";
			att++;
		}
		context.write("params", new Text("a2"));
		data.setClassIndex(data.numAttributes()-1);
		Attribute classAtt = data.classAttribute();
		output=output+"@attribute class {"+classAtt.value(0)+","+classAtt.value(1)+"}\n";
		output=output+"\n@data\n\n";
		br.close();
		context.write("params", new Text("a3"));
		for (int i = 0; i < data.numInstances(); i++) {
			Instance inst = data.get(i);
			for (int j = 0; j < modelList.size(); j++) {
				String model = modelList.get(j);
				String[] params = model.split(" ");
				//System.out.println("\nparams.length="+params.length);
				//System.out.println("\ndata.numAttributes()="+data.numAttributes());
				double value = Double.parseDouble(params[0]);
//				if (params.length == data.numAttributes())
//					System.out.println("\nCorrect num of attributes");
				for (int k = 1; k < data.numAttributes(); k++) {
					value += inst.value(k-1) * Double.parseDouble(params[k]);
				}
				output=output+value+" ";
			}
			
			classAtt = inst.attribute(inst.numAttributes()-1);
			output=output+classAtt.value((int)inst.value(inst.numAttributes()-1))+"\n";
		}
		
		context.write("params", new Text("a4"));
		
		return output;
	}

	public String writefromArray(Context context,FSDataOutputStream out ) throws Exception {

//		context.write("params", new Text("a1"));
		String s="";
		DataSource dt = null;
		Instances data = null;
		// try {
		dt = new DataSource(inputFile);
//		context.write("params", new Text("a2"));
		data = dt.getDataSet();
//		context.write("params", new Text("a3"));
		// } catch (Exception e) {
		// }
//		if(data == null)
//			context.write("params", new Text("data null"));
//		else
//			context.write("params", new Text("data not null"));
		
//		context.write("params", new Text("a4"));
		
		String line;

//		FileWriter fw = new FileWriter(outputFile);
//		context.write("params", new Text("a5"));
//		BufferedWriter bw = new BufferedWriter(fw);
//		context.write("params", new Text("a5"));
//		bw.write(data.relationName()+"2\n\n");
//		s=s+data.relationName()+"2\n\n";
		out.writeBytes("@relation "+data.relationName()+"2\n\n");
//		context.write("params", new Text(data.relationName()+"2\n\n"));
		int att=1;
		int ctr=0;
		while (ctr < modelArray.length) {
//			bw.write("@attribute name"+att+" numeric\n");
//			s=s+"@attribute name"+att+" numeric\n";
			out.writeBytes("@attribute name"+att+" numeric\n");
//			context.write("params", new Text("@attribute name"+att+" numeric\n"));
			att++;
			ctr++;
		}
		data.setClassIndex(data.numAttributes()-1);
		Attribute classAtt = data.classAttribute();
//		bw.write("@attribute class {"+classAtt.value(0)+","+classAtt.value(1)+"}\n");
//		s=s+"@attribute class {"+classAtt.value(0)+","+classAtt.value(1)+"}\n";
		out.writeBytes("@attribute class {"+classAtt.value(0)+","+classAtt.value(1)+"}\n");
//		context.write("params", new Text("@attribute class {"+classAtt.value(0)+","+classAtt.value(1)+"}\n"));
//		bw.write("\n@data\n\n");
//		s=s+"\n@data\n\n";
		out.writeBytes("\n@data\n\n");
//		context.write("params", new Text("\n@data\n\n"));
		
		for (int i = 0; i < data.numInstances(); i++) {
			Instance inst = data.get(i);
			for (int j = 0; j < modelArray.length; j++) {
				String model = modelArray[j];
				String[] params = model.split(" ");
				//System.out.println("\nparams.length="+params.length);
				//System.out.println("\ndata.numAttributes()="+data.numAttributes());
				double value = Double.parseDouble(params[0]);
//				if (params.length == data.numAttributes())
//					System.out.println("\nCorrect num of attributes");
				for (int k = 1; k < data.numAttributes(); k++) {
					value += inst.value(k-1) * Double.parseDouble(params[k]);
				}
//				bw.write(value+" ");
//				s=s+value+" ";
				out.writeBytes(value+" ");
//				context.write("params", new Text(value+" "));
			}
			classAtt = inst.attribute(inst.numAttributes()-1);
//			bw.write(classAtt.value((int)inst.value(inst.numAttributes()-1))+"\n");
//			s=s+classAtt.value((int)inst.value(inst.numAttributes()-1))+"\n";
			out.writeBytes(classAtt.value((int)inst.value(inst.numAttributes()-1))+"\n");
//			context.write("params", new Text(classAtt.value((int)inst.value(inst.numAttributes()-1))+"\n"));
		}

//		context.write(new Text("params"), new Text(s));
		
		
//		bw.close();
		
		return s;
	}	
	
	
	
	public static void main(String[] args) throws Exception {
		FeatureConverter fc = new FeatureConverter(
				"/home/agam/workspace/data/models",
				"/home/agam/workspace/data/trainset-2-exe-split4.arff",
				"/home/agam/workspace/data/output.arff");
//		fc.convert();		
		fc.writefromFile();
	}
}
