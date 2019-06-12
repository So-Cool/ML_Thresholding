ML_Thresholding
===============

Instance-wise thresholding in multi-label classification.

---

* Delicious has been altered m*\'*usica **to** m*A*usica.
* CAL500 split into train:test -> 2:1 ratio
* In eclipse remember to add external jars and arguments to run. The `classpath` is:
    - `src`: `src`,
    - `con`: `org.eclipse.jdt.launching.JRE_CONTAINER/org.eclipse.jdt.internal.debug.ui.launcher.StandardVMType/JavaSE-1.8`,
    - `lib`: `MultilabelThresholding/exlib/mulan.jar`,
    - `lib`: `MultilabelThresholding/exlib/weka.jar` and
    - `output`: `bin`.
* External libraries:
    - Meka version 1.6.2 and
    - Mulan version 1.4.0.
* [Datasets](http://mulan.sourceforge.net/datasets-mlc.html) used for experiments (should go into `MultiLabelThresholding/datasets/*`):
    - [`CAL500`](http://sourceforge.net/projects/mulan/files/datasets/CAL500.rar),
    - [`delicious`](http://sourceforge.net/projects/mulan/files/datasets/delicious.rar),
    - [`emotions`](http://sourceforge.net/projects/mulan/files/datasets/emotions.rar),
    - [`mediamill`](http://sourceforge.net/projects/mulan/files/datasets/mediamill.rar) and
    - [`yeast`](http://sourceforge.net/projects/mulan/files/datasets/yeast.rar).
