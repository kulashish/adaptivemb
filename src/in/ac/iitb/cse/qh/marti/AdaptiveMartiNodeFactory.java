package in.ac.iitb.cse.qh.marti;

import weka.core.Instances;

public class AdaptiveMartiNodeFactory {
	private static Instances instances;

	public static void setInstances(Instances insts) {
		instances = new Instances(insts, 0);
	}

	public static AdaptiveMartiNode createNode() {
		return new AdaptiveMartiNode(instances, instances, instances, null, 0);
	}
}
