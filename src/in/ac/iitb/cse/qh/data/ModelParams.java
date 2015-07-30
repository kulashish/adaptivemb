package in.ac.iitb.cse.qh.data;

import java.util.List;

public class ModelParams {
	private int numParams;
	private boolean optim = false;

	private double[] params;
	private double[] Wparams;
	private List<InputPredictionInstance> predInst;
	private ConfusionMatrix confMatrix;
	
	public boolean isOptim() {
		return optim;
	}

	public void setOptim(boolean optim) {
		this.optim = optim;
	}

	public double[] getParams() {
		return params;
	}

	public void setParams(double[] params) {
		this.params = params;
	}

	public double[] getWParams() {
		return Wparams;
	}

	public void setWParams(double[] Wparams) {
		this.Wparams = Wparams;
	}
	
	public void setConfMatrix(ConfusionMatrix c)
	{
		this.confMatrix = c;
	}
	
	public ConfusionMatrix getConfMatrix()
	{
		return confMatrix;
	}

	public List<InputPredictionInstance> getPredInstances()
	{
		return predInst;
	}
	
	public void setPredInstances(List<InputPredictionInstance> pi)
	{
		predInst=pi;
	}
}
