package engine;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class Thresholders {
	
	private MultiLabelInstances train;
	private List<Integer> labels = new ArrayList<Integer>();
	private List<String> labelsS = new ArrayList<String>();
	
	// labels bounds
	private int bound0;
	private int bound1;
	
	public Thresholders(MultiLabelInstances train) {
		this.train = train;
	}

	public void getLabelCount(){
		int numInstances = train.getNumInstances();
		int classes1 = train.getDataSet().instance(0).numAttributes();
		this.bound1 = classes1;
		int classes0 = classes1 - train.getNumLabels();
		this.bound0 = classes0;
        for ( int instanceIndex = 0; instanceIndex < numInstances; ++instanceIndex ) {
            Instance instance = train.getDataSet().instance(instanceIndex);
            int ones = onesCount(instance, classes0, classes1);
            labels.add( ones );
            labelsS.add(String.valueOf(ones));
        }
	}
	
	private int onesCount(Instance instance, int classes0, int classes1){
		int count = 0;
		for (int i = classes0; i < classes1; ++i)
			if(Integer.valueOf(instance.toString(i)) == 1)
				++count;
		return count;
	}
	
	void alterDataset() throws IOException {
		Instances singleLables = train.getDataSet();
		
		// remove labels
		for (int i = bound0; i < bound1; ++i)
			singleLables.deleteAttributeAt(i);
		
		// add target attribute
		Attribute attr = new Attribute("NumberOfLabels", labelsS);
		
		singleLables.insertAttributeAt(attr, singleLables.numAttributes());
		
		singleLables.setClassIndex(singleLables.numAttributes() - 1);
		
		// Save the file for view
		ArffSaver saver = new ArffSaver();
		saver.setInstances(singleLables);
		saver.setFile(new File("./test.arff"));
//		saver.setDestination(new File(".test.arff"));   // **not** necessary in 3.5.4 and later
		saver.writeBatch();
	}
	
	public List<Integer> getLabels() { return labels; }
	public MultiLabelInstances getTrain() { return train; }
	
}
