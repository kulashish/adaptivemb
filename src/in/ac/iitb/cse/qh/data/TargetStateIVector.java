package in.ac.iitb.cse.qh.data;

import in.ac.iitb.cse.qh.util.MetaConstants;

public class TargetStateIVector extends TargetStateVector {
	public TargetStateIVector(int classIndex) {
		this(MetaConstants.NUMBER_CLASSLABELS, classIndex);
	}

	public TargetStateIVector(int labels, int classIndex) {
		super(labels);
		si[classIndex] = 1.0d;
	}
}
