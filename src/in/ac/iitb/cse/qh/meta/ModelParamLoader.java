package in.ac.iitb.cse.qh.meta;

import in.ac.iitb.cse.qh.data.ModelParams;
import in.ac.iitb.cse.qh.util.MetaConstants;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ModelParamLoader {

	private static final String PARAM_REGEX = "(\\d+)\\s([E\\-\\s.\\d]+)";
	private static final Pattern PARAM_PATTERN = Pattern.compile(PARAM_REGEX);

	public static ModelParams load(BufferedReader reader) throws IOException {
		ModelParams params = null;
		while (!reader.readLine().contains(MetaConstants.IN_MODEL_BEGIN))
			;
		String line = null;
		while (!(line = reader.readLine())
				.equalsIgnoreCase(MetaConstants.IN_MODEL_END))
			params = parseLine(line);

		return params;
	}

	public static ModelParams parseLine(String line) {
		ModelParams params = new ModelParams();
		int num = 0;
		String[] strParams = null;
		double val[] = null;
		Matcher matcher = PARAM_PATTERN.matcher(line);
		if (null != matcher && matcher.find()) {
			num = Integer.parseInt(matcher.group(1));
			System.out.println("Number of params : " + num);
			val = new double[num];
			strParams = matcher.group(2).trim().split("\\s");
			System.out.println("Size of params split : " + strParams.length);
			for (int i = 0; i < num; i++) {
				val[i] = Double.parseDouble(strParams[i]);
			}
			params.setParams(val);
		}
		return params;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		ModelParamLoader.parseLine("2 0.5 0.1");

	}

}
