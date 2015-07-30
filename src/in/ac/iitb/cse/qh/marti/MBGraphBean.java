package in.ac.iitb.cse.qh.marti;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import javax.faces.bean.ManagedBean;
import javax.faces.bean.SessionScoped;

import org.primefaces.event.SelectEvent;
import org.primefaces.event.UnselectEvent;

@ManagedBean(name = "mbgraph", eager = true)
@SessionScoped
public class MBGraphBean implements Serializable {
	private MartiBoost mboost;
	private List<ArrayList<MartiNode>> graphModel;

	public MBGraphBean() {
	}

	public List<ArrayList<MartiNode>> getGraphModel() {
		if (null == graphModel)
			graphModel = new ArrayList<ArrayList<MartiNode>>();
		return graphModel;
	}

	public void setGraphModel(List<ArrayList<MartiNode>> graphModel) {
		this.graphModel = graphModel;
	}

	public void onSelectNodes(SelectEvent event) {
		System.out.println("NODES SELECTION:" + event.getObject());
	}

	public void onUnselectNodes(UnselectEvent event) {
		System.out.println("NODES UNSELECTION:" + event.getObject());
	}

	public void setMarti(MartiBoost mboost) {
		this.mboost = mboost;
	}

	public void showNode() {
		System.out.println("show node called");
	}
}
