# **Digit Recognizer - Lept1**


```python
import sys

MODULE_FULL_PATH = '/home/leptone/Libraries'

sys.path.insert(1, MODULE_FULL_PATH)

from library import *
```


```python
from IPython.display import IFrame

IFrame(src='https://en.wikipedia.org/wiki/handwriting', width=700, height=500)
```





<iframe
    width="700"
    height="500"
    src="https://en.wikipedia.org/wiki/handwriting"
    frameborder="0"
    allowfullscreen
></iframe>




# Load the train data

Firstly, we must load the dataset. We use [Pandas](https://pandas.pydata.org/) . I load the dataset and convert the categorical non numeric features into numeric ones.


```python
train_data = pd.read_csv("train.csv", encoding="utf8")
```


```python
test_data = pd.read_csv("test.csv", encoding="utf8")
```


```python
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(train_data.dtypes)
```

    label       int64
    pixel0      int64
    pixel1      int64
    pixel2      int64
    pixel3      int64
    pixel4      int64
    pixel5      int64
    pixel6      int64
    pixel7      int64
    pixel8      int64
    pixel9      int64
    pixel10     int64
    pixel11     int64
    pixel12     int64
    pixel13     int64
    pixel14     int64
    pixel15     int64
    pixel16     int64
    pixel17     int64
    pixel18     int64
    pixel19     int64
    pixel20     int64
    pixel21     int64
    pixel22     int64
    pixel23     int64
    pixel24     int64
    pixel25     int64
    pixel26     int64
    pixel27     int64
    pixel28     int64
    pixel29     int64
    pixel30     int64
    pixel31     int64
    pixel32     int64
    pixel33     int64
    pixel34     int64
    pixel35     int64
    pixel36     int64
    pixel37     int64
    pixel38     int64
    pixel39     int64
    pixel40     int64
    pixel41     int64
    pixel42     int64
    pixel43     int64
    pixel44     int64
    pixel45     int64
    pixel46     int64
    pixel47     int64
    pixel48     int64
    pixel49     int64
    pixel50     int64
    pixel51     int64
    pixel52     int64
    pixel53     int64
    pixel54     int64
    pixel55     int64
    pixel56     int64
    pixel57     int64
    pixel58     int64
    pixel59     int64
    pixel60     int64
    pixel61     int64
    pixel62     int64
    pixel63     int64
    pixel64     int64
    pixel65     int64
    pixel66     int64
    pixel67     int64
    pixel68     int64
    pixel69     int64
    pixel70     int64
    pixel71     int64
    pixel72     int64
    pixel73     int64
    pixel74     int64
    pixel75     int64
    pixel76     int64
    pixel77     int64
    pixel78     int64
    pixel79     int64
    pixel80     int64
    pixel81     int64
    pixel82     int64
    pixel83     int64
    pixel84     int64
    pixel85     int64
    pixel86     int64
    pixel87     int64
    pixel88     int64
    pixel89     int64
    pixel90     int64
    pixel91     int64
    pixel92     int64
    pixel93     int64
    pixel94     int64
    pixel95     int64
    pixel96     int64
    pixel97     int64
    pixel98     int64
    pixel99     int64
    pixel100    int64
    pixel101    int64
    pixel102    int64
    pixel103    int64
    pixel104    int64
    pixel105    int64
    pixel106    int64
    pixel107    int64
    pixel108    int64
    pixel109    int64
    pixel110    int64
    pixel111    int64
    pixel112    int64
    pixel113    int64
    pixel114    int64
    pixel115    int64
    pixel116    int64
    pixel117    int64
    pixel118    int64
    pixel119    int64
    pixel120    int64
    pixel121    int64
    pixel122    int64
    pixel123    int64
    pixel124    int64
    pixel125    int64
    pixel126    int64
    pixel127    int64
    pixel128    int64
    pixel129    int64
    pixel130    int64
    pixel131    int64
    pixel132    int64
    pixel133    int64
    pixel134    int64
    pixel135    int64
    pixel136    int64
    pixel137    int64
    pixel138    int64
    pixel139    int64
    pixel140    int64
    pixel141    int64
    pixel142    int64
    pixel143    int64
    pixel144    int64
    pixel145    int64
    pixel146    int64
    pixel147    int64
    pixel148    int64
    pixel149    int64
    pixel150    int64
    pixel151    int64
    pixel152    int64
    pixel153    int64
    pixel154    int64
    pixel155    int64
    pixel156    int64
    pixel157    int64
    pixel158    int64
    pixel159    int64
    pixel160    int64
    pixel161    int64
    pixel162    int64
    pixel163    int64
    pixel164    int64
    pixel165    int64
    pixel166    int64
    pixel167    int64
    pixel168    int64
    pixel169    int64
    pixel170    int64
    pixel171    int64
    pixel172    int64
    pixel173    int64
    pixel174    int64
    pixel175    int64
    pixel176    int64
    pixel177    int64
    pixel178    int64
    pixel179    int64
    pixel180    int64
    pixel181    int64
    pixel182    int64
    pixel183    int64
    pixel184    int64
    pixel185    int64
    pixel186    int64
    pixel187    int64
    pixel188    int64
    pixel189    int64
    pixel190    int64
    pixel191    int64
    pixel192    int64
    pixel193    int64
    pixel194    int64
    pixel195    int64
    pixel196    int64
    pixel197    int64
    pixel198    int64
    pixel199    int64
    pixel200    int64
    pixel201    int64
    pixel202    int64
    pixel203    int64
    pixel204    int64
    pixel205    int64
    pixel206    int64
    pixel207    int64
    pixel208    int64
    pixel209    int64
    pixel210    int64
    pixel211    int64
    pixel212    int64
    pixel213    int64
    pixel214    int64
    pixel215    int64
    pixel216    int64
    pixel217    int64
    pixel218    int64
    pixel219    int64
    pixel220    int64
    pixel221    int64
    pixel222    int64
    pixel223    int64
    pixel224    int64
    pixel225    int64
    pixel226    int64
    pixel227    int64
    pixel228    int64
    pixel229    int64
    pixel230    int64
    pixel231    int64
    pixel232    int64
    pixel233    int64
    pixel234    int64
    pixel235    int64
    pixel236    int64
    pixel237    int64
    pixel238    int64
    pixel239    int64
    pixel240    int64
    pixel241    int64
    pixel242    int64
    pixel243    int64
    pixel244    int64
    pixel245    int64
    pixel246    int64
    pixel247    int64
    pixel248    int64
    pixel249    int64
    pixel250    int64
    pixel251    int64
    pixel252    int64
    pixel253    int64
    pixel254    int64
    pixel255    int64
    pixel256    int64
    pixel257    int64
    pixel258    int64
    pixel259    int64
    pixel260    int64
    pixel261    int64
    pixel262    int64
    pixel263    int64
    pixel264    int64
    pixel265    int64
    pixel266    int64
    pixel267    int64
    pixel268    int64
    pixel269    int64
    pixel270    int64
    pixel271    int64
    pixel272    int64
    pixel273    int64
    pixel274    int64
    pixel275    int64
    pixel276    int64
    pixel277    int64
    pixel278    int64
    pixel279    int64
    pixel280    int64
    pixel281    int64
    pixel282    int64
    pixel283    int64
    pixel284    int64
    pixel285    int64
    pixel286    int64
    pixel287    int64
    pixel288    int64
    pixel289    int64
    pixel290    int64
    pixel291    int64
    pixel292    int64
    pixel293    int64
    pixel294    int64
    pixel295    int64
    pixel296    int64
    pixel297    int64
    pixel298    int64
    pixel299    int64
    pixel300    int64
    pixel301    int64
    pixel302    int64
    pixel303    int64
    pixel304    int64
    pixel305    int64
    pixel306    int64
    pixel307    int64
    pixel308    int64
    pixel309    int64
    pixel310    int64
    pixel311    int64
    pixel312    int64
    pixel313    int64
    pixel314    int64
    pixel315    int64
    pixel316    int64
    pixel317    int64
    pixel318    int64
    pixel319    int64
    pixel320    int64
    pixel321    int64
    pixel322    int64
    pixel323    int64
    pixel324    int64
    pixel325    int64
    pixel326    int64
    pixel327    int64
    pixel328    int64
    pixel329    int64
    pixel330    int64
    pixel331    int64
    pixel332    int64
    pixel333    int64
    pixel334    int64
    pixel335    int64
    pixel336    int64
    pixel337    int64
    pixel338    int64
    pixel339    int64
    pixel340    int64
    pixel341    int64
    pixel342    int64
    pixel343    int64
    pixel344    int64
    pixel345    int64
    pixel346    int64
    pixel347    int64
    pixel348    int64
    pixel349    int64
    pixel350    int64
    pixel351    int64
    pixel352    int64
    pixel353    int64
    pixel354    int64
    pixel355    int64
    pixel356    int64
    pixel357    int64
    pixel358    int64
    pixel359    int64
    pixel360    int64
    pixel361    int64
    pixel362    int64
    pixel363    int64
    pixel364    int64
    pixel365    int64
    pixel366    int64
    pixel367    int64
    pixel368    int64
    pixel369    int64
    pixel370    int64
    pixel371    int64
    pixel372    int64
    pixel373    int64
    pixel374    int64
    pixel375    int64
    pixel376    int64
    pixel377    int64
    pixel378    int64
    pixel379    int64
    pixel380    int64
    pixel381    int64
    pixel382    int64
    pixel383    int64
    pixel384    int64
    pixel385    int64
    pixel386    int64
    pixel387    int64
    pixel388    int64
    pixel389    int64
    pixel390    int64
    pixel391    int64
    pixel392    int64
    pixel393    int64
    pixel394    int64
    pixel395    int64
    pixel396    int64
    pixel397    int64
    pixel398    int64
    pixel399    int64
    pixel400    int64
    pixel401    int64
    pixel402    int64
    pixel403    int64
    pixel404    int64
    pixel405    int64
    pixel406    int64
    pixel407    int64
    pixel408    int64
    pixel409    int64
    pixel410    int64
    pixel411    int64
    pixel412    int64
    pixel413    int64
    pixel414    int64
    pixel415    int64
    pixel416    int64
    pixel417    int64
    pixel418    int64
    pixel419    int64
    pixel420    int64
    pixel421    int64
    pixel422    int64
    pixel423    int64
    pixel424    int64
    pixel425    int64
    pixel426    int64
    pixel427    int64
    pixel428    int64
    pixel429    int64
    pixel430    int64
    pixel431    int64
    pixel432    int64
    pixel433    int64
    pixel434    int64
    pixel435    int64
    pixel436    int64
    pixel437    int64
    pixel438    int64
    pixel439    int64
    pixel440    int64
    pixel441    int64
    pixel442    int64
    pixel443    int64
    pixel444    int64
    pixel445    int64
    pixel446    int64
    pixel447    int64
    pixel448    int64
    pixel449    int64
    pixel450    int64
    pixel451    int64
    pixel452    int64
    pixel453    int64
    pixel454    int64
    pixel455    int64
    pixel456    int64
    pixel457    int64
    pixel458    int64
    pixel459    int64
    pixel460    int64
    pixel461    int64
    pixel462    int64
    pixel463    int64
    pixel464    int64
    pixel465    int64
    pixel466    int64
    pixel467    int64
    pixel468    int64
    pixel469    int64
    pixel470    int64
    pixel471    int64
    pixel472    int64
    pixel473    int64
    pixel474    int64
    pixel475    int64
    pixel476    int64
    pixel477    int64
    pixel478    int64
    pixel479    int64
    pixel480    int64
    pixel481    int64
    pixel482    int64
    pixel483    int64
    pixel484    int64
    pixel485    int64
    pixel486    int64
    pixel487    int64
    pixel488    int64
    pixel489    int64
    pixel490    int64
    pixel491    int64
    pixel492    int64
    pixel493    int64
    pixel494    int64
    pixel495    int64
    pixel496    int64
    pixel497    int64
    pixel498    int64
    pixel499    int64
    pixel500    int64
    pixel501    int64
    pixel502    int64
    pixel503    int64
    pixel504    int64
    pixel505    int64
    pixel506    int64
    pixel507    int64
    pixel508    int64
    pixel509    int64
    pixel510    int64
    pixel511    int64
    pixel512    int64
    pixel513    int64
    pixel514    int64
    pixel515    int64
    pixel516    int64
    pixel517    int64
    pixel518    int64
    pixel519    int64
    pixel520    int64
    pixel521    int64
    pixel522    int64
    pixel523    int64
    pixel524    int64
    pixel525    int64
    pixel526    int64
    pixel527    int64
    pixel528    int64
    pixel529    int64
    pixel530    int64
    pixel531    int64
    pixel532    int64
    pixel533    int64
    pixel534    int64
    pixel535    int64
    pixel536    int64
    pixel537    int64
    pixel538    int64
    pixel539    int64
    pixel540    int64
    pixel541    int64
    pixel542    int64
    pixel543    int64
    pixel544    int64
    pixel545    int64
    pixel546    int64
    pixel547    int64
    pixel548    int64
    pixel549    int64
    pixel550    int64
    pixel551    int64
    pixel552    int64
    pixel553    int64
    pixel554    int64
    pixel555    int64
    pixel556    int64
    pixel557    int64
    pixel558    int64
    pixel559    int64
    pixel560    int64
    pixel561    int64
    pixel562    int64
    pixel563    int64
    pixel564    int64
    pixel565    int64
    pixel566    int64
    pixel567    int64
    pixel568    int64
    pixel569    int64
    pixel570    int64
    pixel571    int64
    pixel572    int64
    pixel573    int64
    pixel574    int64
    pixel575    int64
    pixel576    int64
    pixel577    int64
    pixel578    int64
    pixel579    int64
    pixel580    int64
    pixel581    int64
    pixel582    int64
    pixel583    int64
    pixel584    int64
    pixel585    int64
    pixel586    int64
    pixel587    int64
    pixel588    int64
    pixel589    int64
    pixel590    int64
    pixel591    int64
    pixel592    int64
    pixel593    int64
    pixel594    int64
    pixel595    int64
    pixel596    int64
    pixel597    int64
    pixel598    int64
    pixel599    int64
    pixel600    int64
    pixel601    int64
    pixel602    int64
    pixel603    int64
    pixel604    int64
    pixel605    int64
    pixel606    int64
    pixel607    int64
    pixel608    int64
    pixel609    int64
    pixel610    int64
    pixel611    int64
    pixel612    int64
    pixel613    int64
    pixel614    int64
    pixel615    int64
    pixel616    int64
    pixel617    int64
    pixel618    int64
    pixel619    int64
    pixel620    int64
    pixel621    int64
    pixel622    int64
    pixel623    int64
    pixel624    int64
    pixel625    int64
    pixel626    int64
    pixel627    int64
    pixel628    int64
    pixel629    int64
    pixel630    int64
    pixel631    int64
    pixel632    int64
    pixel633    int64
    pixel634    int64
    pixel635    int64
    pixel636    int64
    pixel637    int64
    pixel638    int64
    pixel639    int64
    pixel640    int64
    pixel641    int64
    pixel642    int64
    pixel643    int64
    pixel644    int64
    pixel645    int64
    pixel646    int64
    pixel647    int64
    pixel648    int64
    pixel649    int64
    pixel650    int64
    pixel651    int64
    pixel652    int64
    pixel653    int64
    pixel654    int64
    pixel655    int64
    pixel656    int64
    pixel657    int64
    pixel658    int64
    pixel659    int64
    pixel660    int64
    pixel661    int64
    pixel662    int64
    pixel663    int64
    pixel664    int64
    pixel665    int64
    pixel666    int64
    pixel667    int64
    pixel668    int64
    pixel669    int64
    pixel670    int64
    pixel671    int64
    pixel672    int64
    pixel673    int64
    pixel674    int64
    pixel675    int64
    pixel676    int64
    pixel677    int64
    pixel678    int64
    pixel679    int64
    pixel680    int64
    pixel681    int64
    pixel682    int64
    pixel683    int64
    pixel684    int64
    pixel685    int64
    pixel686    int64
    pixel687    int64
    pixel688    int64
    pixel689    int64
    pixel690    int64
    pixel691    int64
    pixel692    int64
    pixel693    int64
    pixel694    int64
    pixel695    int64
    pixel696    int64
    pixel697    int64
    pixel698    int64
    pixel699    int64
    pixel700    int64
    pixel701    int64
    pixel702    int64
    pixel703    int64
    pixel704    int64
    pixel705    int64
    pixel706    int64
    pixel707    int64
    pixel708    int64
    pixel709    int64
    pixel710    int64
    pixel711    int64
    pixel712    int64
    pixel713    int64
    pixel714    int64
    pixel715    int64
    pixel716    int64
    pixel717    int64
    pixel718    int64
    pixel719    int64
    pixel720    int64
    pixel721    int64
    pixel722    int64
    pixel723    int64
    pixel724    int64
    pixel725    int64
    pixel726    int64
    pixel727    int64
    pixel728    int64
    pixel729    int64
    pixel730    int64
    pixel731    int64
    pixel732    int64
    pixel733    int64
    pixel734    int64
    pixel735    int64
    pixel736    int64
    pixel737    int64
    pixel738    int64
    pixel739    int64
    pixel740    int64
    pixel741    int64
    pixel742    int64
    pixel743    int64
    pixel744    int64
    pixel745    int64
    pixel746    int64
    pixel747    int64
    pixel748    int64
    pixel749    int64
    pixel750    int64
    pixel751    int64
    pixel752    int64
    pixel753    int64
    pixel754    int64
    pixel755    int64
    pixel756    int64
    pixel757    int64
    pixel758    int64
    pixel759    int64
    pixel760    int64
    pixel761    int64
    pixel762    int64
    pixel763    int64
    pixel764    int64
    pixel765    int64
    pixel766    int64
    pixel767    int64
    pixel768    int64
    pixel769    int64
    pixel770    int64
    pixel771    int64
    pixel772    int64
    pixel773    int64
    pixel774    int64
    pixel775    int64
    pixel776    int64
    pixel777    int64
    pixel778    int64
    pixel779    int64
    pixel780    int64
    pixel781    int64
    pixel782    int64
    pixel783    int64
    dtype: object



```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>



how many samples?


```python
train_data.pixel0.shape
```




    (42000,)




```python
test_data.pixel0.shape
```




    (28000,)




```python
sns.countplot(y='label', data=train_data)
```




    <AxesSubplot:xlabel='count', ylabel='label'>




    
![png](output_12_1.png)
    


## Extract the target


```python
y = train_data['label']
```


```python
y.head()
```




    0    1
    1    0
    2    1
    3    4
    4    0
    Name: label, dtype: int64




```python
y.shape
```




    (42000,)



# Features analysis

how much NaN value are there?


```python
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print (train_data.isnull().sum())
```

    label       0
    pixel0      0
    pixel1      0
    pixel2      0
    pixel3      0
    pixel4      0
    pixel5      0
    pixel6      0
    pixel7      0
    pixel8      0
    pixel9      0
    pixel10     0
    pixel11     0
    pixel12     0
    pixel13     0
    pixel14     0
    pixel15     0
    pixel16     0
    pixel17     0
    pixel18     0
    pixel19     0
    pixel20     0
    pixel21     0
    pixel22     0
    pixel23     0
    pixel24     0
    pixel25     0
    pixel26     0
    pixel27     0
    pixel28     0
    pixel29     0
    pixel30     0
    pixel31     0
    pixel32     0
    pixel33     0
    pixel34     0
    pixel35     0
    pixel36     0
    pixel37     0
    pixel38     0
    pixel39     0
    pixel40     0
    pixel41     0
    pixel42     0
    pixel43     0
    pixel44     0
    pixel45     0
    pixel46     0
    pixel47     0
    pixel48     0
    pixel49     0
    pixel50     0
    pixel51     0
    pixel52     0
    pixel53     0
    pixel54     0
    pixel55     0
    pixel56     0
    pixel57     0
    pixel58     0
    pixel59     0
    pixel60     0
    pixel61     0
    pixel62     0
    pixel63     0
    pixel64     0
    pixel65     0
    pixel66     0
    pixel67     0
    pixel68     0
    pixel69     0
    pixel70     0
    pixel71     0
    pixel72     0
    pixel73     0
    pixel74     0
    pixel75     0
    pixel76     0
    pixel77     0
    pixel78     0
    pixel79     0
    pixel80     0
    pixel81     0
    pixel82     0
    pixel83     0
    pixel84     0
    pixel85     0
    pixel86     0
    pixel87     0
    pixel88     0
    pixel89     0
    pixel90     0
    pixel91     0
    pixel92     0
    pixel93     0
    pixel94     0
    pixel95     0
    pixel96     0
    pixel97     0
    pixel98     0
    pixel99     0
    pixel100    0
    pixel101    0
    pixel102    0
    pixel103    0
    pixel104    0
    pixel105    0
    pixel106    0
    pixel107    0
    pixel108    0
    pixel109    0
    pixel110    0
    pixel111    0
    pixel112    0
    pixel113    0
    pixel114    0
    pixel115    0
    pixel116    0
    pixel117    0
    pixel118    0
    pixel119    0
    pixel120    0
    pixel121    0
    pixel122    0
    pixel123    0
    pixel124    0
    pixel125    0
    pixel126    0
    pixel127    0
    pixel128    0
    pixel129    0
    pixel130    0
    pixel131    0
    pixel132    0
    pixel133    0
    pixel134    0
    pixel135    0
    pixel136    0
    pixel137    0
    pixel138    0
    pixel139    0
    pixel140    0
    pixel141    0
    pixel142    0
    pixel143    0
    pixel144    0
    pixel145    0
    pixel146    0
    pixel147    0
    pixel148    0
    pixel149    0
    pixel150    0
    pixel151    0
    pixel152    0
    pixel153    0
    pixel154    0
    pixel155    0
    pixel156    0
    pixel157    0
    pixel158    0
    pixel159    0
    pixel160    0
    pixel161    0
    pixel162    0
    pixel163    0
    pixel164    0
    pixel165    0
    pixel166    0
    pixel167    0
    pixel168    0
    pixel169    0
    pixel170    0
    pixel171    0
    pixel172    0
    pixel173    0
    pixel174    0
    pixel175    0
    pixel176    0
    pixel177    0
    pixel178    0
    pixel179    0
    pixel180    0
    pixel181    0
    pixel182    0
    pixel183    0
    pixel184    0
    pixel185    0
    pixel186    0
    pixel187    0
    pixel188    0
    pixel189    0
    pixel190    0
    pixel191    0
    pixel192    0
    pixel193    0
    pixel194    0
    pixel195    0
    pixel196    0
    pixel197    0
    pixel198    0
    pixel199    0
    pixel200    0
    pixel201    0
    pixel202    0
    pixel203    0
    pixel204    0
    pixel205    0
    pixel206    0
    pixel207    0
    pixel208    0
    pixel209    0
    pixel210    0
    pixel211    0
    pixel212    0
    pixel213    0
    pixel214    0
    pixel215    0
    pixel216    0
    pixel217    0
    pixel218    0
    pixel219    0
    pixel220    0
    pixel221    0
    pixel222    0
    pixel223    0
    pixel224    0
    pixel225    0
    pixel226    0
    pixel227    0
    pixel228    0
    pixel229    0
    pixel230    0
    pixel231    0
    pixel232    0
    pixel233    0
    pixel234    0
    pixel235    0
    pixel236    0
    pixel237    0
    pixel238    0
    pixel239    0
    pixel240    0
    pixel241    0
    pixel242    0
    pixel243    0
    pixel244    0
    pixel245    0
    pixel246    0
    pixel247    0
    pixel248    0
    pixel249    0
    pixel250    0
    pixel251    0
    pixel252    0
    pixel253    0
    pixel254    0
    pixel255    0
    pixel256    0
    pixel257    0
    pixel258    0
    pixel259    0
    pixel260    0
    pixel261    0
    pixel262    0
    pixel263    0
    pixel264    0
    pixel265    0
    pixel266    0
    pixel267    0
    pixel268    0
    pixel269    0
    pixel270    0
    pixel271    0
    pixel272    0
    pixel273    0
    pixel274    0
    pixel275    0
    pixel276    0
    pixel277    0
    pixel278    0
    pixel279    0
    pixel280    0
    pixel281    0
    pixel282    0
    pixel283    0
    pixel284    0
    pixel285    0
    pixel286    0
    pixel287    0
    pixel288    0
    pixel289    0
    pixel290    0
    pixel291    0
    pixel292    0
    pixel293    0
    pixel294    0
    pixel295    0
    pixel296    0
    pixel297    0
    pixel298    0
    pixel299    0
    pixel300    0
    pixel301    0
    pixel302    0
    pixel303    0
    pixel304    0
    pixel305    0
    pixel306    0
    pixel307    0
    pixel308    0
    pixel309    0
    pixel310    0
    pixel311    0
    pixel312    0
    pixel313    0
    pixel314    0
    pixel315    0
    pixel316    0
    pixel317    0
    pixel318    0
    pixel319    0
    pixel320    0
    pixel321    0
    pixel322    0
    pixel323    0
    pixel324    0
    pixel325    0
    pixel326    0
    pixel327    0
    pixel328    0
    pixel329    0
    pixel330    0
    pixel331    0
    pixel332    0
    pixel333    0
    pixel334    0
    pixel335    0
    pixel336    0
    pixel337    0
    pixel338    0
    pixel339    0
    pixel340    0
    pixel341    0
    pixel342    0
    pixel343    0
    pixel344    0
    pixel345    0
    pixel346    0
    pixel347    0
    pixel348    0
    pixel349    0
    pixel350    0
    pixel351    0
    pixel352    0
    pixel353    0
    pixel354    0
    pixel355    0
    pixel356    0
    pixel357    0
    pixel358    0
    pixel359    0
    pixel360    0
    pixel361    0
    pixel362    0
    pixel363    0
    pixel364    0
    pixel365    0
    pixel366    0
    pixel367    0
    pixel368    0
    pixel369    0
    pixel370    0
    pixel371    0
    pixel372    0
    pixel373    0
    pixel374    0
    pixel375    0
    pixel376    0
    pixel377    0
    pixel378    0
    pixel379    0
    pixel380    0
    pixel381    0
    pixel382    0
    pixel383    0
    pixel384    0
    pixel385    0
    pixel386    0
    pixel387    0
    pixel388    0
    pixel389    0
    pixel390    0
    pixel391    0
    pixel392    0
    pixel393    0
    pixel394    0
    pixel395    0
    pixel396    0
    pixel397    0
    pixel398    0
    pixel399    0
    pixel400    0
    pixel401    0
    pixel402    0
    pixel403    0
    pixel404    0
    pixel405    0
    pixel406    0
    pixel407    0
    pixel408    0
    pixel409    0
    pixel410    0
    pixel411    0
    pixel412    0
    pixel413    0
    pixel414    0
    pixel415    0
    pixel416    0
    pixel417    0
    pixel418    0
    pixel419    0
    pixel420    0
    pixel421    0
    pixel422    0
    pixel423    0
    pixel424    0
    pixel425    0
    pixel426    0
    pixel427    0
    pixel428    0
    pixel429    0
    pixel430    0
    pixel431    0
    pixel432    0
    pixel433    0
    pixel434    0
    pixel435    0
    pixel436    0
    pixel437    0
    pixel438    0
    pixel439    0
    pixel440    0
    pixel441    0
    pixel442    0
    pixel443    0
    pixel444    0
    pixel445    0
    pixel446    0
    pixel447    0
    pixel448    0
    pixel449    0
    pixel450    0
    pixel451    0
    pixel452    0
    pixel453    0
    pixel454    0
    pixel455    0
    pixel456    0
    pixel457    0
    pixel458    0
    pixel459    0
    pixel460    0
    pixel461    0
    pixel462    0
    pixel463    0
    pixel464    0
    pixel465    0
    pixel466    0
    pixel467    0
    pixel468    0
    pixel469    0
    pixel470    0
    pixel471    0
    pixel472    0
    pixel473    0
    pixel474    0
    pixel475    0
    pixel476    0
    pixel477    0
    pixel478    0
    pixel479    0
    pixel480    0
    pixel481    0
    pixel482    0
    pixel483    0
    pixel484    0
    pixel485    0
    pixel486    0
    pixel487    0
    pixel488    0
    pixel489    0
    pixel490    0
    pixel491    0
    pixel492    0
    pixel493    0
    pixel494    0
    pixel495    0
    pixel496    0
    pixel497    0
    pixel498    0
    pixel499    0
    pixel500    0
    pixel501    0
    pixel502    0
    pixel503    0
    pixel504    0
    pixel505    0
    pixel506    0
    pixel507    0
    pixel508    0
    pixel509    0
    pixel510    0
    pixel511    0
    pixel512    0
    pixel513    0
    pixel514    0
    pixel515    0
    pixel516    0
    pixel517    0
    pixel518    0
    pixel519    0
    pixel520    0
    pixel521    0
    pixel522    0
    pixel523    0
    pixel524    0
    pixel525    0
    pixel526    0
    pixel527    0
    pixel528    0
    pixel529    0
    pixel530    0
    pixel531    0
    pixel532    0
    pixel533    0
    pixel534    0
    pixel535    0
    pixel536    0
    pixel537    0
    pixel538    0
    pixel539    0
    pixel540    0
    pixel541    0
    pixel542    0
    pixel543    0
    pixel544    0
    pixel545    0
    pixel546    0
    pixel547    0
    pixel548    0
    pixel549    0
    pixel550    0
    pixel551    0
    pixel552    0
    pixel553    0
    pixel554    0
    pixel555    0
    pixel556    0
    pixel557    0
    pixel558    0
    pixel559    0
    pixel560    0
    pixel561    0
    pixel562    0
    pixel563    0
    pixel564    0
    pixel565    0
    pixel566    0
    pixel567    0
    pixel568    0
    pixel569    0
    pixel570    0
    pixel571    0
    pixel572    0
    pixel573    0
    pixel574    0
    pixel575    0
    pixel576    0
    pixel577    0
    pixel578    0
    pixel579    0
    pixel580    0
    pixel581    0
    pixel582    0
    pixel583    0
    pixel584    0
    pixel585    0
    pixel586    0
    pixel587    0
    pixel588    0
    pixel589    0
    pixel590    0
    pixel591    0
    pixel592    0
    pixel593    0
    pixel594    0
    pixel595    0
    pixel596    0
    pixel597    0
    pixel598    0
    pixel599    0
    pixel600    0
    pixel601    0
    pixel602    0
    pixel603    0
    pixel604    0
    pixel605    0
    pixel606    0
    pixel607    0
    pixel608    0
    pixel609    0
    pixel610    0
    pixel611    0
    pixel612    0
    pixel613    0
    pixel614    0
    pixel615    0
    pixel616    0
    pixel617    0
    pixel618    0
    pixel619    0
    pixel620    0
    pixel621    0
    pixel622    0
    pixel623    0
    pixel624    0
    pixel625    0
    pixel626    0
    pixel627    0
    pixel628    0
    pixel629    0
    pixel630    0
    pixel631    0
    pixel632    0
    pixel633    0
    pixel634    0
    pixel635    0
    pixel636    0
    pixel637    0
    pixel638    0
    pixel639    0
    pixel640    0
    pixel641    0
    pixel642    0
    pixel643    0
    pixel644    0
    pixel645    0
    pixel646    0
    pixel647    0
    pixel648    0
    pixel649    0
    pixel650    0
    pixel651    0
    pixel652    0
    pixel653    0
    pixel654    0
    pixel655    0
    pixel656    0
    pixel657    0
    pixel658    0
    pixel659    0
    pixel660    0
    pixel661    0
    pixel662    0
    pixel663    0
    pixel664    0
    pixel665    0
    pixel666    0
    pixel667    0
    pixel668    0
    pixel669    0
    pixel670    0
    pixel671    0
    pixel672    0
    pixel673    0
    pixel674    0
    pixel675    0
    pixel676    0
    pixel677    0
    pixel678    0
    pixel679    0
    pixel680    0
    pixel681    0
    pixel682    0
    pixel683    0
    pixel684    0
    pixel685    0
    pixel686    0
    pixel687    0
    pixel688    0
    pixel689    0
    pixel690    0
    pixel691    0
    pixel692    0
    pixel693    0
    pixel694    0
    pixel695    0
    pixel696    0
    pixel697    0
    pixel698    0
    pixel699    0
    pixel700    0
    pixel701    0
    pixel702    0
    pixel703    0
    pixel704    0
    pixel705    0
    pixel706    0
    pixel707    0
    pixel708    0
    pixel709    0
    pixel710    0
    pixel711    0
    pixel712    0
    pixel713    0
    pixel714    0
    pixel715    0
    pixel716    0
    pixel717    0
    pixel718    0
    pixel719    0
    pixel720    0
    pixel721    0
    pixel722    0
    pixel723    0
    pixel724    0
    pixel725    0
    pixel726    0
    pixel727    0
    pixel728    0
    pixel729    0
    pixel730    0
    pixel731    0
    pixel732    0
    pixel733    0
    pixel734    0
    pixel735    0
    pixel736    0
    pixel737    0
    pixel738    0
    pixel739    0
    pixel740    0
    pixel741    0
    pixel742    0
    pixel743    0
    pixel744    0
    pixel745    0
    pixel746    0
    pixel747    0
    pixel748    0
    pixel749    0
    pixel750    0
    pixel751    0
    pixel752    0
    pixel753    0
    pixel754    0
    pixel755    0
    pixel756    0
    pixel757    0
    pixel758    0
    pixel759    0
    pixel760    0
    pixel761    0
    pixel762    0
    pixel763    0
    pixel764    0
    pixel765    0
    pixel766    0
    pixel767    0
    pixel768    0
    pixel769    0
    pixel770    0
    pixel771    0
    pixel772    0
    pixel773    0
    pixel774    0
    pixel775    0
    pixel776    0
    pixel777    0
    pixel778    0
    pixel779    0
    pixel780    0
    pixel781    0
    pixel782    0
    pixel783    0
    dtype: int64



```python
X=train_data.drop(columns=['label'])
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 784 columns</p>
</div>



## SVD & PCA

First nomalize data


```python
from sklearn.preprocessing import StandardScaler, Normalizer
scaler = Normalizer()
X_scal=scaler.fit_transform(X)
```


```python
test_scal=scaler.transform(test_data)
```

iterate Truncated SVD varying the number of components to find variance 90%


```python
from sklearn.decomposition import PCA, TruncatedSVD
variance=[]
for component in range(5,110,5):
    svd = TruncatedSVD(n_components=component, n_iter=7, random_state=42)
    svd.fit(X_scal)
    variance.append(svd.explained_variance_ratio_.sum())
    #print (component)

```


```python
import matplotlib.pyplot as plt

plt.plot(np.arange(5,110,5),variance)
plt.plot([0,105],[0.90,0.90])
plt.plot(90,0.9,'ro')
for i in variance:
    if i >= 0.9:
        print('# of components for variance 0.9 = ',variance.index(i)*5)
        break
```

    # of components for variance 0.9 =  90



    
![png](output_28_1.png)
    


now calculate PCA the found number of components


```python
from sklearn.decomposition import PCA,TruncatedSVD
pca = PCA(n_components=90)
X_new=pca.fit_transform(X_scal)
```


```python
test_new=pca.transform(test_scal)
```

How many features?


```python
X_new.shape
```




    (42000, 90)




```python
X_tr=pca.inverse_transform(X_new)
```


```python
v = np.array(X_tr).reshape(-1, 28, 28, 1)/255

fig, ax = plt.subplots(1, 5, figsize=(15,8))
for i in range(5):
    ax[i].imshow(v[i], cmap='binary')
```


    
![png](output_35_0.png)
    


## Logistic Regression


```python
X_train, X_test, y_train, y_test = train_test_split(X_new, y,random_state=42,test_size=0.2)
```


```python
rf=training(X_train,y_train,p_name='digit',model='Random Forest')
```

    Fitting 5 folds for each of 100 candidates, totalling 500 fits


    [Parallel(n_jobs=-3)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-3)]: Done  33 tasks      | elapsed:    6.0s
    [Parallel(n_jobs=-3)]: Done 154 tasks      | elapsed:  1.7min
    [Parallel(n_jobs=-3)]: Done 357 tasks      | elapsed:  7.3min
    [Parallel(n_jobs=-3)]: Done 500 out of 500 | elapsed: 11.5min finished



```python
target_names=['0','1','2','3','4','5','6','7','8','9']
```


```python
predic = rf.predict(X_test)
confm = confusion_matrix(y_test, predic,normalize='true')
plot_confusion_matrix(confm,target_names)
```


    
![png](output_40_0.png)
    



```python
best_rf=rf.best_estimator_
best_rf.fit(X_new,y)
scores=cross_val_score(best_rf, X_new,y, cv=5,scoring=make_scorer(accuracy_score))
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

    Accuracy: 0.94 (+/- 0.00)


## Convolutional Neural Network

I tried to use CNN with PCA with very bad results. It's reasonable, because CNN need to recognize pattern.

Let's summarize. The input of our neural network is a tensor with width=28, height=28 and depth=1 where $28 \times 28$ is the dimension of the image and $ 1 $ the colour channel.  

### First step = Convolution
Instead  of  a  full  connection,  it  is  a  good  idea  to  look  for  local  regions  in  the  picture  instead  of  in  the  whole  image. Another  assumption  for  simplification,  is  to  keep  the  local  connection  weights  fixed  for  the  entire  neurons  of  the  next  layer.  This  will  connect  the  neighbor  neurons  in  the  next  layer  with  exactly  the  same  weight  to  the  local  region  of  the  previous  layer.
<img src="convolution_schematic.gif" width="750" align="center">

In this case, the yellow matrix is the filter. 

### Second step = Non linearity
For many years, sigmoid and tanh were the most popular non-linearity. The following figure  shows   the   common   types   of   nonlinearity. 
<img src="nnlin.png" width="750" align="center">

### Third step = Pooling
Spatial Pooling (also called subsampling or downsampling) reduces the dimensionality of each feature map but retains the most important information. Spatial Pooling can be of different types: Max, Average, Sum etc.
Below an example of Max Pooling operation on a Rectified Feature map (obtained after convolution + ReLU operation) by using a 2×2 window.
<img src="pooling.png" width="750" align="center">

### Fourth step = Fully connected
The Fully Connected layer is a traditional Multi Layer Perceptron that uses a softmax activation function in the output layer. The term “Fully Connected” implies that every neuron in the previous layer is connected to every neuron on the next layer. The output from the convolutional and pooling layers represent high-level features of the input image. The purpose of the Fully Connected layer is to use these features for classifying the input image into various classes based on the training dataset. 
<img src="fullconn.png" width="750" align="center">


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, MaxPooling2D,Conv2D,Flatten
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
```


```python
X.shape
```




    (42000, 784)




```python
X=X.to_numpy()
X = X.reshape(-1, 28, 28, 1) 

```


```python
X.shape
```




    (42000, 28, 28, 1)




```python
test_data_cnn=test_data.to_numpy()
test_data_cnn=test_data_cnn.reshape(-1, 28, 28, 1) 
```


```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10))


```


```python
model.summary()
```

    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_18 (Conv2D)           (None, 26, 26, 32)        320       
    _________________________________________________________________
    max_pooling2d_12 (MaxPooling (None, 13, 13, 32)        0         
    _________________________________________________________________
    conv2d_19 (Conv2D)           (None, 11, 11, 64)        18496     
    _________________________________________________________________
    max_pooling2d_13 (MaxPooling (None, 5, 5, 64)          0         
    _________________________________________________________________
    conv2d_20 (Conv2D)           (None, 3, 3, 64)          36928     
    _________________________________________________________________
    flatten_6 (Flatten)          (None, 576)               0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 64)                36928     
    _________________________________________________________________
    dense_13 (Dense)             (None, 10)                650       
    =================================================================
    Total params: 93,322
    Trainable params: 93,322
    Non-trainable params: 0
    _________________________________________________________________


### first run


```python
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42,test_size=0.2)
```


```python
model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))
```

    Epoch 1/10
    1050/1050 [==============================] - 41s 39ms/step - loss: 0.4643 - accuracy: 0.8986 - val_loss: 0.1221 - val_accuracy: 0.9640
    Epoch 2/10
    1050/1050 [==============================] - 44s 42ms/step - loss: 0.1072 - accuracy: 0.9664 - val_loss: 0.0791 - val_accuracy: 0.9746
    Epoch 3/10
    1050/1050 [==============================] - 42s 40ms/step - loss: 0.0748 - accuracy: 0.9766 - val_loss: 0.0715 - val_accuracy: 0.9769
    Epoch 4/10
    1050/1050 [==============================] - 40s 38ms/step - loss: 0.0601 - accuracy: 0.9800 - val_loss: 0.0857 - val_accuracy: 0.9746
    Epoch 5/10
    1050/1050 [==============================] - 40s 39ms/step - loss: 0.0511 - accuracy: 0.9837 - val_loss: 0.1157 - val_accuracy: 0.9673
    Epoch 6/10
    1050/1050 [==============================] - 42s 40ms/step - loss: 0.0469 - accuracy: 0.9849 - val_loss: 0.0981 - val_accuracy: 0.9724
    Epoch 7/10
    1050/1050 [==============================] - 41s 39ms/step - loss: 0.0340 - accuracy: 0.9893 - val_loss: 0.0635 - val_accuracy: 0.9818
    Epoch 8/10
    1050/1050 [==============================] - 42s 40ms/step - loss: 0.0306 - accuracy: 0.9901 - val_loss: 0.0695 - val_accuracy: 0.9813
    Epoch 9/10
    1050/1050 [==============================] - 41s 39ms/step - loss: 0.0291 - accuracy: 0.9906 - val_loss: 0.0593 - val_accuracy: 0.9843
    Epoch 10/10
    1050/1050 [==============================] - 42s 40ms/step - loss: 0.0242 - accuracy: 0.9923 - val_loss: 0.0638 - val_accuracy: 0.9831


### second run


```python
model.compile(optimizer='Adagrad',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))
```

    Epoch 1/10
    1050/1050 [==============================] - 43s 41ms/step - loss: 0.0107 - accuracy: 0.9966 - val_loss: 0.0495 - val_accuracy: 0.9875
    Epoch 2/10
    1050/1050 [==============================] - 41s 39ms/step - loss: 0.0053 - accuracy: 0.9985 - val_loss: 0.0485 - val_accuracy: 0.9876
    Epoch 3/10
    1050/1050 [==============================] - 43s 41ms/step - loss: 0.0042 - accuracy: 0.9990 - val_loss: 0.0494 - val_accuracy: 0.9873
    Epoch 4/10
    1050/1050 [==============================] - 41s 39ms/step - loss: 0.0037 - accuracy: 0.9992 - val_loss: 0.0499 - val_accuracy: 0.9871
    Epoch 5/10
    1050/1050 [==============================] - 41s 39ms/step - loss: 0.0032 - accuracy: 0.9994 - val_loss: 0.0507 - val_accuracy: 0.9870
    Epoch 6/10
    1050/1050 [==============================] - 42s 40ms/step - loss: 0.0029 - accuracy: 0.9995 - val_loss: 0.0507 - val_accuracy: 0.9873
    Epoch 7/10
    1050/1050 [==============================] - 42s 40ms/step - loss: 0.0027 - accuracy: 0.9995 - val_loss: 0.0514 - val_accuracy: 0.9874
    Epoch 8/10
    1050/1050 [==============================] - 41s 39ms/step - loss: 0.0025 - accuracy: 0.9996 - val_loss: 0.0518 - val_accuracy: 0.9873
    Epoch 9/10
    1050/1050 [==============================] - 42s 40ms/step - loss: 0.0023 - accuracy: 0.9997 - val_loss: 0.0519 - val_accuracy: 0.9873
    Epoch 10/10
    1050/1050 [==============================] - 43s 41ms/step - loss: 0.0021 - accuracy: 0.9997 - val_loss: 0.0523 - val_accuracy: 0.9873


### third run on full dataset


```python
model.compile(optimizer='RMSprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X, y, epochs=10)
```

    Epoch 1/10
    1313/1313 [==============================] - 51s 39ms/step - loss: 0.0970 - accuracy: 0.9774
    Epoch 2/10
    1313/1313 [==============================] - 50s 38ms/step - loss: 0.0840 - accuracy: 0.9808
    Epoch 3/10
    1313/1313 [==============================] - 49s 37ms/step - loss: 0.0863 - accuracy: 0.9816
    Epoch 4/10
    1313/1313 [==============================] - 49s 38ms/step - loss: 0.0910 - accuracy: 0.9829
    Epoch 5/10
    1313/1313 [==============================] - 49s 38ms/step - loss: 0.0822 - accuracy: 0.9850
    Epoch 6/10
    1313/1313 [==============================] - 49s 37ms/step - loss: 0.0724 - accuracy: 0.9865
    Epoch 7/10
    1313/1313 [==============================] - 47s 36ms/step - loss: 0.0895 - accuracy: 0.9865
    Epoch 8/10
    1313/1313 [==============================] - 49s 37ms/step - loss: 0.0906 - accuracy: 0.9867
    Epoch 9/10
    1313/1313 [==============================] - 51s 39ms/step - loss: 0.0958 - accuracy: 0.9875
    Epoch 10/10
    1313/1313 [==============================] - 49s 38ms/step - loss: 0.1048 - accuracy: 0.9883



```python
from sklearn import metrics
predic = model.predict(X_test)
print(metrics.classification_report(y_test, np.argmax(predic,axis=1)))
```

                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99       816
               1       0.99      1.00      1.00       909
               2       0.99      0.99      0.99       846
               3       1.00      0.98      0.99       937
               4       0.98      0.99      0.99       839
               5       1.00      0.98      0.99       702
               6       0.99      0.99      0.99       785
               7       0.97      1.00      0.99       893
               8       0.98      0.99      0.99       835
               9       0.99      0.96      0.98       838
    
        accuracy                           0.99      8400
       macro avg       0.99      0.99      0.99      8400
    weighted avg       0.99      0.99      0.99      8400
    


# Prepare the solution


```python
#solution for other models
y_pred=best_rf.predict(test_new)
iid=range(1,int((test_new[:,0].shape)[0])+1)
solution = pd.DataFrame({'ImageId':iid,'Label':y_pred})
print(solution.head(10))
solution.to_csv("solution_rf.csv", index=False)
```

       ImageId  Label
    0        1      2
    1        2      0
    2        3      9
    3        4      7
    4        5      2
    5        6      7
    6        7      0
    7        8      3
    8        9      0
    9       10      3



```python
!kaggle competitions submit -c digit-recognizer -f solution_rf.csv -m "RF+PCA90"
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.12 / client 1.5.10)
    100%|████████████████████████████████████████| 208k/208k [00:02<00:00, 77.9kB/s]
    Successfully submitted to Digit Recognizer


```python
##solution for cnn
y_pred=model.predict(test_data_cnn)
iid=range(1,int((test_data_cnn[:,0].shape)[0])+1)
solution = pd.DataFrame({'ImageId':iid,'Label':np.argmax(y_pred,axis = 1)})
print(solution.head(10))
solution.to_csv("solution_cnn.csv", index=False)
```

       ImageId  Label
    0        1      2
    1        2      0
    2        3      9
    3        4      0
    4        5      3
    5        6      7
    6        7      0
    7        8      3
    8        9      0
    9       10      3



```python
!kaggle competitions submit -c digit-recognizer -f solution_cnn.csv -m "CNN"
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.12 / client 1.5.10)
    100%|████████████████████████████████████████| 208k/208k [00:04<00:00, 53.1kB/s]
    403 - Your team has used its submission allowance (5 of 5). This resets at midnight UTC (9.4 hours from now).



```python

```
