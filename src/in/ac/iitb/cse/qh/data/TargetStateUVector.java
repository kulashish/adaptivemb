package in.ac.iitb.cse.qh.data;

import in.ac.iitb.cse.qh.util.MetaConstants;

public class TargetStateUVector extends TargetStateVector {
	public TargetStateUVector(int classIndex) {
		this(MetaConstants.NUMBER_CLASSLABELS, classIndex);
	}

	public TargetStateUVector(int labels, int classIndex) {
		super(labels);
		for (int i = 0; i < numLabels; i++)
			if (i != classIndex)
				si[i] = 1 / (numLabels - 1);
	}
}
