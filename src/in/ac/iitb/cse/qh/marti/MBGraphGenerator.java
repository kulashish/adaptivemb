package in.ac.iitb.cse.qh.marti;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import in.ac.iitb.cse.qh.util.BeanFinder;
import in.ac.iitb.cse.qh.util.MetaConstants;

import org.primefaces.model.mindmap.DefaultMindmapNode;
import org.primefaces.model.mindmap.MindmapNode;

public class MBGraphGenerator {
	private static final Logger LOGGER = Logger
			.getLogger(MBGraphGenerator.class.getName());

	static {
		LOGGER.setLevel(Level.INFO);
	}

	public static MBGraphBean generateGraph(MartiBoost mboost) {
		MBGraphBean mBean = (MBGraphBean) BeanFinder
				.findBean(MetaConstants.BEAN_MBOOST_GRAPH);
		if (null == mBean)
			mBean = new MBGraphBean();

		MindmapNode mbRoot = new DefaultMindmapNode("V-00", "Root");
		mBean.setRoot(mbRoot);

		List<MindmapNode> graphNodes = new ArrayList<MindmapNode>();
		graphNodes.add(mbRoot);
		for (int level = 1; level < mboost.getNumLevels(); level++) {
			LOGGER.log(Level.INFO, "Adding level " + level);
			graphNodes = addLevel(level, mboost.getRoot(), graphNodes);
		}
		return mBean;
	}

	public static List<MindmapNode> addLevel(int level, MartiNode node,
			List<MindmapNode> graphLevelNodes) {
		MindmapNode graphNode = null;
		MindmapNode leftGraphNode = null;
		MindmapNode rightGraphNode = null;
		List<MindmapNode> newGraphLevelNodes = new ArrayList<MindmapNode>();
		for (int i = 0; i < graphLevelNodes.size(); i++) {
			graphNode = graphLevelNodes.get(i);
			if (i == 0) {
				leftGraphNode = new DefaultMindmapNode("V" + level + i, "V"
						+ level + i);
				graphNode.addNode(leftGraphNode);
				newGraphLevelNodes.add(leftGraphNode);
			} else
				graphNode.addNode(rightGraphNode);

			rightGraphNode = new DefaultMindmapNode("V" + level + (i + 1), "V"
					+ level + (i + 1));
			graphNode.addNode(rightGraphNode);
			newGraphLevelNodes.add(rightGraphNode);
		}
		return newGraphLevelNodes;
	}
}
