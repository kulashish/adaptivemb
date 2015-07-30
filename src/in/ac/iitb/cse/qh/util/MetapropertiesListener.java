package in.ac.iitb.cse.qh.util;

import javax.servlet.ServletContextEvent;
import javax.servlet.ServletContextListener;

public class MetapropertiesListener implements ServletContextListener {

	@Override
	public void contextDestroyed(ServletContextEvent arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public void contextInitialized(ServletContextEvent event) {
		System.out.println("Metaproperties Listener called");
		String val = event.getServletContext().getInitParameter("root_path");
		System.out.println("value=" + val);
		MetaConstants.ROOT_PATH = val;
		MetaConstants.configure();
	}

}
