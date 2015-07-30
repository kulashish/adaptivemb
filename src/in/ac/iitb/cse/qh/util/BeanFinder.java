package in.ac.iitb.cse.qh.util;

import javax.faces.context.FacesContext;

public class BeanFinder {
	@SuppressWarnings("unchecked")
	public static <T> T findBean(String beanName) {
		T bean = null;
		FacesContext context = FacesContext.getCurrentInstance();
		if (null != context)
			bean = (T) context.getExternalContext().getSessionMap()
					.get(beanName);
		return bean;
	}
}
