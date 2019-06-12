package engine;

import mulan.data.MultiLabelInstances;
/*
 * To-Do
 * split TEST/TRAINING: CAL500 dataset
 * fix the 4th multilabel classifer: RANKING BY PAIRWISE COMPARISON 
 */

public class Engine {

    // private static int CVfolds = 2;
    private static String evaluationMeasure = "Hamming Loss";

    // save model to files
    private static Boolean saveModels = false;

    public static void main(String[] args) throws Exception {

        // check for data-set argument
        // in order: training header test???
        if (args.length != 3) {
            System.out.println("Lacking dataset and header as attributes.\nExpecting `training.set header.file test.set`\n");
            System.exit(1);
        }

        // load data-set
        System.out.println("Reading training dataset...");
        MultiLabelInstances training = new MultiLabelInstances(args[0], args[1]);
        System.out.println("Reading test dataset...");
        MultiLabelInstances test = new MultiLabelInstances(args[2], args[1]);

        Thresholders thr = new Thresholders(training);
        thr.getLabelCount();
        thr.alterDataset();
        if( training.equals(thr.getTrain()) )
            System.out.println( "They are equal" );
        System.exit(0);

        // use data-set with *CVfolds*-folds cross-validation
        Learners learn = new Learners(training, test, saveModels);
        String[] filename = args[1].split("\\.")[0].split("\\/");
        learn.learn(filename[filename.length-1]);
        learn.evaluate(evaluationMeasure);
        learn.threshold();

        // Gather HammingLoss results and print them
        System.out.println(learn.toString());

        // Exit message
        System.out.println("Done!");
    }

}
