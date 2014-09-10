package engine;

import mulan.data.MultiLabelInstances;
// datasets/CAL500/CAL500.arff datasets/CAL500/CAL500.xml

public class Engine {

    public static void main(String[] args) throws Exception {
    	
    	// check for data-set argument
    	if (args.length != 2) {
	        System.out.println("Lacking dataset and header as attributes.\n");
	        System.exit(1);
	     }

		// load data-set
    	System.out.println("Reading dataset...");
    	MultiLabelInstances dataset = new MultiLabelInstances(args[0], args[1]);
    	
    	// use data-set with 10-folds cross-validation
    	Learners learn = new Learners(dataset, 2);
    	learn.evaluate();
    	
    	
    	
    	
    	
    	//  old weka
//    	BufferedReader reader = new BufferedReader( new FileReader(args[0]) );
//    	Instances data = new Instances( reader );
//    	reader.close();
    	//  new weka
//    	DataSource source = new DataSource("/some/where/data.arff");
//    	Instances data = source.getDataSet();
    	
    	// setting class attribute
//    	data.setClassIndex(data.numAttributes() - 1);
	 
    	// running classifiers
//    	MultiLabelLearner.runClassifier(new TestClassifier(),args);
    	
	    System.out.println("Done!");
	}
	
}
