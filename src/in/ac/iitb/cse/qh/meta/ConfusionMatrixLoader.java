package in.ac.iitb.cse.qh.meta;

import in.ac.iitb.cse.qh.data.ConfusionMatrix;
import in.ac.iitb.cse.qh.util.MetaConstants;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ConfusionMatrixLoader {
	private static final String CONF_REGEX = "(\\d+)\\s*";
	private static final Pattern CONF_PATTERN = Pattern.compile(CONF_REGEX);

	public static ConfusionMatrix load(BufferedReader reader)
			throws IOException {
		ConfusionMatrix matrix = null;
		while (!reader.readLine().contains(MetaConstants.IN_CONFUSION_BEGIN))
			;
		String line = null;
		while (!(line = reader.readLine())
				.equalsIgnoreCase(MetaConstants.IN_CONFUSION_END))
			matrix = parseLine(line);

		return matrix;
	}

	public static ConfusionMatrix parseLine(String line) {
		ConfusionMatrix matrix = new ConfusionMatrix();
		int num = matrix.getNumLables();
		int val[][] = new int[num][num];
		Matcher matcher = CONF_PATTERN.matcher(line);
		if (null != matcher) {
			for (int i = 0; i < num; i++)
				for (int j = 0; j < num; j++){
					matcher.find();
					val[i][j] = Integer.parseInt(matcher.group().trim());
				}

			matrix.setMatrix(val);
		}
		
		return matrix;
	}

	public static void main(String[] args) {
		ConfusionMatrixLoader.parseLine("50 30 40 20");
	}
}
