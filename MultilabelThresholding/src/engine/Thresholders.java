package engine;

import java.util.ArrayList;
import java.util.List;

import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;

public class Thresholders {
	
	private MultiLabelInstances train;
	private List<Integer> labels = new ArrayList<Integer>();
	
	public Thresholders(MultiLabelInstances train) {
		this.train = train;
	}

	public void getLabelCount(){
		int numInstances = train.getNumInstances();
		int classes1 = train.getDataSet().instance(0).numAttributes();
		int classes0 = classes1 - train.getNumLabels();
        for ( int instanceIndex = 0; instanceIndex < numInstances; ++instanceIndex ) {
            Instance instance = train.getDataSet().instance(instanceIndex);
            int ones = onesCount(instance, classes0, classes1);
            System.out.println("labels: " + ones);
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
	
	void alterDataset() {
		Instances singleLables = train.getDataSet();
		int numInstances = singleLables.numInstances();
		for ( int instanceIndex = 0; instanceIndex < numInstances; ++instanceIndex ) {
            Instance instance = singleLables.instance(instanceIndex);
		}
	}
	
	public List<Integer> getLabels() { return labels; }
	public MultiLabelInstances getTrain() { return train; }
	
}
