package in.ac.iitb.cse.qh.meta;

import in.ac.iitb.cse.qh.data.ConfusionMatrix;
import in.ac.iitb.cse.qh.data.CurrentState;
import in.ac.iitb.cse.qh.data.InputData;
import in.ac.iitb.cse.qh.data.TargetState;

public class DefensiveOptimizer extends Optimizer {

	public DefensiveOptimizer(InputData in, CurrentState curr,
			TargetState target, ClassifierProxy classifier) {
		super(in, curr, target, classifier);
	}

	protected boolean optimized(double[] theta) throws Exception {
		ConfusionMatrix newConf = newData.getConfMatrix();
//		newConf.display();
		int[][] nc = newConf.getMatrix();
		int[][] c = data.getConfMatrix().getMatrix();
		int[][] b = data.getBiasMatrix().getMatrix();
		boolean blnOptim = true;
		for (int i = 0; i < numLabels; i++)
			for (int j = 0; j < numLabels; j++)
				if (b[i][j] > c[i][j])
					blnOptim = nc[i][j] > c[i][j];
				else if (b[i][j] < c[i][j])
					blnOptim = nc[i][j] < c[i][j];

		if (blnOptim) {
//			try {
//				System.out.println("Serializing data after optimization...");
//				newData.serialize(MetaConstants.OPTIMIZED_FILE_PATH);
//			} catch (IOException e) {
//				e.printStackTrace();
//			}
			data.setConfMatrix(newConf);
			data.setPredInstances(newData.getPredInstances());
		}

		return blnOptim;
	}
}
