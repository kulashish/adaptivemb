package in.ac.iitb.cse.qh.data;

import in.ac.iitb.cse.qh.util.MetaConstants;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.richfaces.event.FileUploadEvent;
import org.richfaces.model.UploadedFile;

public class FileUploadBean implements Serializable {
	private List<String> files;

	public FileUploadBean() throws IOException {
		File dir = new File(MetaConstants.UPLOAD_PATH);
		if (!dir.exists())
			dir.mkdir();
		File f = new File(MetaConstants.UPLOAD_PATH + "files");
		if (!f.exists())
			f.createNewFile();
		BufferedReader reader = new BufferedReader(new FileReader(f));
		if (null == files)
			files = new ArrayList<String>();
		String file = null;
		while (null != (file = reader.readLine()))
			files.add(file);
		reader.close();
	}

	public void listener(FileUploadEvent event) throws Exception {
		UploadedFile item = event.getUploadedFile();
		if (null == files)
			files = new ArrayList<String>();
		files.add(item.getName());
		File f = new File(MetaConstants.UPLOAD_PATH + item.getName());
		if (!f.exists())
			f.createNewFile();
		FileOutputStream fos = new FileOutputStream(f);
		fos.write(item.getData());
		fos.close();

		BufferedWriter writer = new BufferedWriter(new FileWriter(
				MetaConstants.UPLOAD_PATH + "files", true));
		writer.write(item.getName());
		writer.newLine();
		writer.close();
	}

	public List<String> getFiles() {
		return files;
	}

}
