package in.ac.iitb.cse.qh.marti;

import in.ac.iitb.cse.qh.data.ConfusionMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instance;

public class AdaptiveMartiLevel {
	private static final Logger LOGGER = Logger
			.getLogger(AdaptiveMartiLevel.class.getName());

	private int levelNumber;
	private List<AdaptiveMartiNode> nodes;
	private AdaptiveMartiLevel nextLevel;
	private double levelAdvantage = 0.5d;
	private boolean finalLevel;

	public AdaptiveMartiLevel(int level) {
		levelNumber = level;
		nodes = new ArrayList<AdaptiveMartiNode>();
	}

	public void addLevel() {
		nextLevel = new AdaptiveMartiLevel(levelNumber + 1);
	}

	public void addNode(AdaptiveMartiNode node) {
		if (nodes == null)
			nodes = new ArrayList<AdaptiveMartiNode>();
		node.setLevel(this);
		nodes.add(node);
	}

	public boolean isFinalLevel() {
		return finalLevel;
	}

	public void setFinalLevel(boolean finalLevel) {
		this.finalLevel = finalLevel;
	}

	public int getLevelNumber() {
		return levelNumber;
	}

	public void setLevelNumber(int levelNumber) {
		this.levelNumber = levelNumber;
	}

	public AdaptiveMartiLevel getNextLevel() {
		if (nextLevel == null)
			addLevel();
		return nextLevel;
	}

	public void setNextLevel(AdaptiveMartiLevel nextLevel) {
		this.nextLevel = nextLevel;
	}

	public List<AdaptiveMartiNode> getNodes() {
		return nodes;
	}

	public void build() throws Exception {
		for (AdaptiveMartiNode node : nodes) {
			node.build();
			// level advantage is the minimum node-level advantage at that level
			if (!node.isFrozen()
					&& node.getOutData().trainAdvantage < levelAdvantage)
				levelAdvantage = node.getOutData().trainAdvantage;
		}
		// System.out.println("Level advantage: " + levelAdvantage);
		for (AdaptiveMartiNode node : nodes)
			if (!node.isFrozen())
				node.route(levelAdvantage);
	}

	public void routeInstance(int nextLevelIndex, Instance instance, short iType) {
		AdaptiveMartiNode node = getNextLevel().getNode(nextLevelIndex);
		if (node == null) {
			if (iType == 2) {
				LOGGER.log(Level.SEVERE,
						"Test instance being routed to a node that does not exist");
				node = getNextLevel().getNearestNode(nextLevelIndex);
			} else {
				node = AdaptiveMartiNodeFactory.createNode();
				node.setNumber(nextLevelIndex);
				node.setBeta(nextLevelIndex * levelAdvantage / 2);
				getNextLevel().addNode(node);
			}
		}
		node.addInstance(instance, iType);
	}

	private AdaptiveMartiNode getNearestNode(int nodeIndex) {
		int dist = 9999;
		AdaptiveMartiNode nearNode = null;
		for (AdaptiveMartiNode node : nodes)
			if (Math.abs(node.getNumber() - nodeIndex) < dist) {
				dist = Math.abs(node.getNumber() - nodeIndex);
				nearNode = node;
			}
		return nearNode;
	}

	public boolean nodePresent(int nextLevelIndex) {
		return getNextLevel().getNode(nextLevelIndex) != null;
	}

	private AdaptiveMartiNode getNode(int nextLevelIndex) {
		AdaptiveMartiNode found = null;
		for (AdaptiveMartiNode node : getNodes())
			if (node.getNumber() == nextLevelIndex) {
				found = node;
				break;
			}
		return found;
	}

	public void evaluateLevel(boolean blnFinal) {
		LOGGER.log(Level.INFO, "Number of nodes:" + nodes.size());
		for (AdaptiveMartiNode node : nodes)
			// if (blnFinal || node.isFrozen())
			node.display();
	}

	public void classifyRouteTestInstances(ConfusionMatrix gb_matrix)
			throws Exception {
		for (AdaptiveMartiNode node : nodes) {
			LOGGER.log(Level.INFO, "Level: " + node.getLevel().levelNumber
					+ " Node: " + node.getNumber());
			if (isFinalLevel())
				node.classifyFinalLevelTestInstances();
			else
				node.classifyRouteTestInstances(levelAdvantage);
			if (node.isFrozen() || isFinalLevel())
				gb_matrix.addMatrix(node.getOutData().testConfusionMatrix);
			node.getOutData().testConfusionMatrix.display();
		}
	}
}
