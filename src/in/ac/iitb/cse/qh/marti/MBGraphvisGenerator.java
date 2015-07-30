package in.ac.iitb.cse.qh.marti;

import in.ac.iitb.cse.qh.util.BeanFinder;
import in.ac.iitb.cse.qh.util.MetaConstants;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class MBGraphvisGenerator {
	private static final Logger LOGGER = Logger
			.getLogger(MBGraphvisGenerator.class.getName());

	static {
		LOGGER.setLevel(Level.INFO);
	}

	public static MBGraphBean generateGraph(MartiBoost mboost) {
		MBGraphBean mBean = (MBGraphBean) BeanFinder
				.findBean(MetaConstants.BEAN_MBOOST_GRAPH);
		if (null == mBean)
			mBean = new MBGraphBean();

		mBean.setMarti(mboost);
		List<ArrayList<MartiNode>> model = mBean.getGraphModel();
		ArrayList<MartiNode> rootLevel = new ArrayList<MartiNode>();
		rootLevel.add(mboost.getRoot());
		model.add(rootLevel);
		MartiNode node = mboost.getRoot();
		ArrayList<MartiNode> levelNodes = null;

		for (int level = 1; level < mboost.getNumLevels(); level++) {
			LOGGER.log(Level.INFO, "Adding level " + level);
			levelNodes = addLevel(node);
			model.add(levelNodes);
			node = levelNodes.get(0);
		}
		return mBean;
	}

	public static ArrayList<MartiNode> addLevel(MartiNode node) {
		ArrayList<MartiNode> levelNodes = new ArrayList<MartiNode>();
		MartiNode listNode = node.leftNode;
		while (listNode != null) {
			levelNodes.add(listNode);
			listNode = listNode.getNextNode();
		}
		return levelNodes;
	}
}
