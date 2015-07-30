package in.ac.iitb.cse.qh.meta;

import in.ac.iitb.cse.qh.data.InputPredictionInstance;
import in.ac.iitb.cse.qh.util.MetaConstants;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PredictionsLoader {
	// <true label> <pred label> <class score>...
	private static final int NUM_GROUPS = 3;
	private static final String PRED_REGEX = "(\\d+)\\s(\\d+)([E\\-\\s.\\d]+)";
	private static final Pattern PRED_PATTERN = Pattern.compile(PRED_REGEX);

	public static List<InputPredictionInstance> load(BufferedReader reader) throws IOException {
		List<InputPredictionInstance> instances = new ArrayList<InputPredictionInstance>();
		while (!reader.readLine().contains(MetaConstants.IN_PRED_BEGIN))
			;
		String line = null;
		while (!(line = reader.readLine())
				.equalsIgnoreCase(MetaConstants.IN_PRED_END))
			instances.add(parsePrediction(line));
		
		return instances;
	}

	public static InputPredictionInstance parsePrediction(String line) {
		InputPredictionInstance instance = null;
		double[] scores = null;
		String[] strScores = null;
		Matcher matcher = PRED_PATTERN.matcher(line);
		if (null != matcher && matcher.find()
				&& matcher.groupCount() == NUM_GROUPS) {
			strScores = matcher.group(3).trim().split("\\s");
//			System.out.println(strScores.length);
			scores = new double[strScores.length];

			for (int i = 0; i < strScores.length; i++)
				scores[i] = Double.valueOf(strScores[i]);

			instance = new InputPredictionInstance(Integer.parseInt(matcher
					.group(1)), Integer.parseInt(matcher.group(2)), scores);
		}
		return instance;
	}

	public static void main(String[] args) {
		PredictionsLoader.parsePrediction("1 1 0.8 0.2");
	}

}
