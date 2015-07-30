package in.ac.iitb.cse.qh.data;

import in.ac.iitb.cse.qh.util.BeanFinder;
import in.ac.iitb.cse.qh.util.MetaConstants;

import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import javax.faces.application.FacesMessage;
import javax.faces.context.FacesContext;

import org.primefaces.event.ItemSelectEvent;
import org.primefaces.model.DefaultStreamedContent;
import org.primefaces.model.StreamedContent;
import org.primefaces.model.chart.CartesianChartModel;
import org.primefaces.model.chart.ChartSeries;

public class MetaChartBean {
	private CartesianChartModel model;
	private int iteration;
	private ChartSeries data;
	private String message;
	private boolean selected;
	private List<InputData> modelsList;
	private StreamedContent file;

	public MetaChartBean() {
		model = new CartesianChartModel();
		modelsList = new ArrayList<InputData>();
		data = new ChartSeries();
		data.setLabel("False Positives");
		iteration = 0;
		model.addSeries(data);
		System.out.println("Chart object : " + this);
		selected = false;
	}

	public void addModel(InputData data) {
		modelsList.add(data);
	}

	public void addData(double div) {
		selected = false;
		System.out.println("Adding to the chart " + div);
		if (null != data)
			data.getData().put(++iteration, div);
	}

	public CartesianChartModel getModel() {
		return model;
	}

	public void itemSelect(ItemSelectEvent event) throws FileNotFoundException,
			IOException {
		System.out.println("Received event!");
		FacesMessage msg = new FacesMessage(FacesMessage.SEVERITY_INFO,
				"Item selected", "Item Index: " + event.getItemIndex()
						+ ", Series Index:" + event.getSeriesIndex());

		FacesContext.getCurrentInstance().addMessage(null, msg);
		message = "Item Index: " + event.getItemIndex() + ", Series Index:"
				+ event.getSeriesIndex();
		System.out.println(message);

		String strModel = null;
		InputData dataBean = (InputData) BeanFinder
				.findBean(MetaConstants.BEAN_INPUTDATA);
		InputData selectedModel = null;
		try {
			selectedModel = modelsList.get(event.getItemIndex());
			dataBean.copyFrom(selectedModel);
			strModel = selectedModel.serialize();
		} catch (IOException e) {
			dataBean.setError(true);
			dataBean.setErrorLog(e);
		} catch (Exception e) {
			dataBean.setError(true);
			dataBean.setErrorLog(e);
		}
		if (null != strModel) {
			InputStream stream = new ByteArrayInputStream(strModel.getBytes());
			file = new DefaultStreamedContent(stream, "text/plain",
					"downloaded_model.dat");

			ObjectOutputStream oos = new ObjectOutputStream(
					new FileOutputStream(MetaConstants.MODEL_DOWNLOAD_PATH
							+ "nodeModel.model"));
			oos.writeObject(selectedModel.getProxy().getClassifier());
			oos.flush();
			oos.close();
		}
		selected = true;
	}

	public String getMessage() {
		return message;
	}

	public StreamedContent getFile() {
		return file;
	}

	public void reset() {
		iteration = 0;
		data.getData().clear();
		modelsList.clear();
	}

	public boolean isSelected() {
		return selected;
	}

}
