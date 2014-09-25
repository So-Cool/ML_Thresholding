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
	private Instances thrSet;
	private Instances thrSetLiteral;
	private List<Integer> labels = new ArrayList<Integer>();
	
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
		for (int i = bound1-1; i > bound0-1; --i)
			singleLables.deleteAttributeAt(i);
		
		// get a copy for text attribute
		Instances thrSetLiteral = new Instances(singleLables);
		
		// get nominal counter
		List<String> values = new ArrayList<String>();
		for( int i = 0; i < 100; ++i ) {
			values.add( String.valueOf(i) );
		}
		
		// add target attribute
		singleLables.insertAttributeAt(new Attribute("NumberOfLabels"), singleLables.numAttributes());
		thrSetLiteral.insertAttributeAt(new Attribute("NumberOfLabels", values), thrSetLiteral.numAttributes());
		
		if (singleLables.numInstances() == labels.size()){
			for (int i = 0; i < labels.size(); ++i) {
				thrSetLiteral.instance(i).setValue(thrSetLiteral.numAttributes() - 1, labels.get(i));
				singleLables.instance(i).setValue(singleLables.numAttributes() - 1, labels.get(i));
			}
		} else {
			System.err.println(" incorrect number of elements!");
			System.exit(1);
		}
		
		singleLables.setClassIndex(singleLables.numAttributes() - 1);
		thrSetLiteral.setClassIndex(thrSetLiteral.numAttributes() - 1);
		
		this.thrSet = singleLables;
		this.thrSetLiteral = thrSetLiteral;
		
//		/*
		// Save the file for view
		ArffSaver saver = new ArffSaver();
		saver.setInstances(singleLables);
		saver.setFile(new File("./test.arff"));
//		saver.setDestination(new File(".test.arff"));   // **not** necessary in 3.5.4 and later
		saver.writeBatch();
//		*/
	}
	
	public void trainThresholders() {
		// train models for thresholding
		
		// regression
		
		// multi-class classification
	}
	
	public List<Integer> getLabels() { return labels; }
	public MultiLabelInstances getTrain() { return train; }
	public Instances getThrSet() { return thrSet; }
	
}
