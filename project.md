<div dir="rtl">

# דוח פרויקט: חיזוי מחלות לב

מגישים: יונתן גבאי 211847215 ונועה הוניגשטיין 329808554

> **הערה:** מודל <span dir="ltr">Random Forest</span> ומודל <span dir="ltr">SVM</span> (מודלים חמישי ושישי) נוספו לאחר הצגת הפרויקט.

---

## 1. על מאגר הנתונים

בפרויקט הזה עבדנו עם <a dir="ltr" href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">Heart Disease Dataset</a> מתוך <span dir="ltr">Kaggle</span>. המאגר כולל נתונים רפואיים של מטופלים, והמטרה היא לחזות אם קיימת מחלת לב: <span dir="ltr">1 = יש מחלה, 0 = אין מחלה</span>.

בנתונים יש 13 מאפיינים רפואיים עיקריים, למשל גיל, כולסטרול, לחץ דם, סוג כאב חזה, קצב לב מקסימלי ועוד.

| עמודה | תיאור |
|-------|--------|
| <code dir="ltr">age</code> | גיל |
| <code dir="ltr">sex</code> | מין <span dir="ltr">(1=זכר, 0=נקבה)</span> |
| <code dir="ltr">cp</code> | סוג כאב חזה <span dir="ltr">(0-3)</span> |
| <code dir="ltr">trestbps</code> | לחץ דם במנוחה <span dir="ltr">(mm Hg)</span> |
| <code dir="ltr">chol</code> | כולסטרול בדם <span dir="ltr">(mg/dl)</span> |
| <code dir="ltr">fbs</code> | סוכר בצום <span dir="ltr">(&gt; 120 mg/dl)</span> |
| <code dir="ltr">restecg</code> | תוצאת <span dir="ltr">ECG</span> במנוחה |
| <code dir="ltr">thalach</code> | קצב לב מקסימלי |
| <code dir="ltr">exang</code> | אנגינה בעקבות מאמץ |
| <code dir="ltr">oldpeak</code> | ירידת <span dir="ltr">ST</span> במאמץ |
| <code dir="ltr">slope</code> | שיפוע מקטע <span dir="ltr">ST</span> |
| <code dir="ltr">ca</code> | מספר כלי דם ראשיים |
| <code dir="ltr">thal</code> | תלסמיה |

---

## 2. מה רצינו לבדוק

במהלך העבודה שאלנו שלוש שאלות מרכזיות:

1. האם אפשר לחזות מחלת לב על סמך הנתונים הרפואיים?
2. איזה מודל עובד הכי טוב על המאגר הזה?
3. אילו מאפיינים משפיעים הכי הרבה על התחזית?

---

## 3. עיבוד מקדים של הנתונים

### 3.1 הסרת כפילויות

אחד הדברים הראשונים שראינו הוא שיש במאגר שורות כפולות. זה חשוב, כי אם אותה שורה מופיעה גם באימון וגם בבדיקה, נוצר מצב של <span dir="ltr">Data Leakage</span> — כלומר, המודל בעצם כבר "פגש" את הדוגמה קודם.

כדי לבדוק את ההשפעה של זה, הרצנו את המודלים גם עם כפילויות וגם בלי כפילויות. ראינו שהפער הכי בולט היה ב-<span dir="ltr">1-NN</span>: עם כפילויות הוא הגיע ל-<span dir="ltr">1.000</span> דיוק בבדיקה, אבל בלי כפילויות ירד ל-<span dir="ltr">0.770</span>. גם ברשת הנוירונים ראינו פער גדול, מ-<span dir="ltr">1.000</span> ל-<span dir="ltr">0.754</span>. זה חיזק את ההבנה שהכפילויות מנפחות את התוצאה, ולכן מהשלב הזה והלאה עבדנו רק עם נתונים נקיים.

### 3.2 חלוקה לסט אימון וסט בדיקה

אחרי הניקוי חילקנו את הנתונים ל-<span dir="ltr">80%</span> אימון ו-<span dir="ltr">20%</span> בדיקה, בעזרת <span dir="ltr">Stratified Split</span>. בחרנו בחלוקה הזו כדי לשמור על יחס דומה בין המחלקות גם בסט האימון וגם בסט הבדיקה.

בפועל מימשנו את זה בעצמנו ב-<span dir="ltr">preprocessing.py</span>, ובדקנו שהיחסים באמת נשמרו בעזרת <code dir="ltr">np.bincount(y_train)</code> ו-<code dir="ltr">np.bincount(y_test)</code>. בסוף קיבלנו <span dir="ltr">241</span> דוגמאות אימון ו-<span dir="ltr">61</span> דוגמאות בדיקה, עם איזון מחלקות של <span dir="ltr">[110, 131]</span> באימון ו-<span dir="ltr">[28, 33]</span> בבדיקה.

### 3.3 נרמול הנתונים

בחלק מהמודלים, במיוחד כאלה שמבוססים על מרחק, חשוב מאוד שהמאפיינים יהיו באותה סקאלה. לכן מימשנו <code dir="ltr">StandardScaler</code> משלנו ב-<span dir="ltr">preprocessing.py</span>.

הנרמול נעשה לפי הנוסחה:

$$X_{\text{scaled}} = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

נקודה חשובה: את ה-<span dir="ltr">Scaler</span> אימנו רק על סט האימון, ואז הפעלנו אותו על סט הבדיקה. כלומר השתמשנו ב-<code dir="ltr">.fit(X_train)</code> ורק אחר כך ב-<code dir="ltr">.transform(X_test)</code>. עשינו את זה כדי לא להכניס מידע מסט הבדיקה לשלב האימון.

הנרמול היה חשוב במיוחד עבור <span dir="ltr">k-NN</span> ועבור הרשת הנוירונים. בפועל, <span dir="ltr">k-NN (k=5)</span> השתפר מ-<span dir="ltr">0.705</span> דיוק בדיקה בלי נרמול ל-<span dir="ltr">0.869</span> עם נרמול, והרשת הנוירונים השתפרה מ-<span dir="ltr">0.541</span> ל-<span dir="ltr">0.754</span>. לעומת זאת, עץ ההחלטה נשאר על <span dir="ltr">0.770</span> גם עם וגם בלי נרמול, ו-<span dir="ltr">AdaBoost</span> נשאר על <span dir="ltr">0.902</span> בשתי הגרסאות.

### 3.4 אימות צולב

כדי לבחור היפרפרמטרים בצורה אמינה, מימשנו <span dir="ltr">Stratified K-Fold CV</span> עם <span dir="ltr">k=5</span> ב-<span dir="ltr">evaluation.py</span>.

בכל <span dir="ltr">fold</span> אימנו מחדש את ה-<span dir="ltr">Scaler</span> רק על חלק האימון של אותו <span dir="ltr">fold</span>. זה היה חשוב כדי למנוע גם כאן <span dir="ltr">Data Leakage</span>.

---

## 4. מודל ראשון: <span dir="ltr">k-NN</span>

### 4.1 איך המודל עובד

הרעיון ב-<span dir="ltr">k-NN</span> פשוט: עבור כל דוגמה חדשה, המודל מחפש את <span dir="ltr">k</span> השכנים הכי קרובים בנתוני האימון, ומחזיר את מחלקת הרוב שלהם.

את חישוב המרחקים מימשנו בצורה וקטורית ב-<span dir="ltr">NumPy</span>, בעזרת הזהות:

$$\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2a \cdot b$$

כך אפשר לחשב מרחקים מהר בלי לעבור בלולאות על כל דוגמה.

### 4.2 בחירת הערך של <span dir="ltr">k</span>

בדקנו כמה ערכים אפשריים של <span dir="ltr">k</span>:

<code dir="ltr">[1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 61, 71, 81, 91, 111, 131, 151, 181, 211]</code>

בחרנו את הערך שנתן את תוצאת ה-<span dir="ltr">CV</span> הטובה ביותר. במחברת התקבל <span dir="ltr">best k = 11</span>, עם דיוק <span dir="ltr">CV</span> של <span dir="ltr">0.809</span>.

הרעיון היה לבדוק גם ערכים קטנים מאוד, שעלולים לגרום ל-<span dir="ltr">overfitting</span>, וגם ערכים גדולים מאוד, שעלולים לגרום ל-<span dir="ltr">underfitting</span>.

### 4.3 מה ראינו

כשהסתכלנו על גרף של <span dir="ltr">Train Error</span> מול <span dir="ltr">Test Error</span>, היה אפשר לראות את התבנית הצפויה:

- כש-<span dir="ltr">k=1</span>, שגיאת האימון כמעט אפסית, אבל שגיאת הבדיקה גבוהה יותר.
- כש-<span dir="ltr">k</span> גדול מדי, המודל כבר נהיה פשוט מדי.
- באמצע מתקבל האיזון הטוב ביותר.

עבור המודל הזה השתמשנו בנתונים המנורמלים: <code dir="ltr">X_train_s</code> ו-<code dir="ltr">X_test_s</code>.

---

## 5. מודל שני: עץ החלטה

### 5.1 איך המודל עובד

מימשנו עץ החלטה מאפס בצורה רקורסיבית. בכל צומת העץ בוחר את הפיצול שנותן את השיפור הכי גדול לפי <span dir="ltr">Information Gain</span>:

$$\text{Gain}(S, f, t) = H(S) - \frac{|S_L|}{|S|} H(S_L) - \frac{|S_R|}{|S|} H(S_R)$$

בדקנו ספים אפשריים בין ערכים ייחודיים סמוכים של כל מאפיין, ובנוסף חישבנו גם <code dir="ltr">feature_importances_</code> לפי התרומה של כל פיצול לאורך העץ.

### 5.2 בחירת היפרפרמטרים

הרצנו <span dir="ltr">5-fold CV</span> על כל הצירופים של:

- <code dir="ltr">max_depth</code>: <code dir="ltr">[3, 4, 5, 6, 7, 8]</code>
- <code dir="ltr">min_samples_split</code>: <code dir="ltr">[10, 12, 15, 18, 20, 25, 30]</code>
- <code dir="ltr">min_samples_leaf</code>: <code dir="ltr">[3, 4, 5, 6, 7, 8]</code>

בסוף, השילוב הטוב ביותר היה <code dir="ltr">max_depth=5</code>, <code dir="ltr">min_samples_split=15</code>, <code dir="ltr">min_samples_leaf=5</code>, עם דיוק <span dir="ltr">CV</span> של <span dir="ltr">0.813 ± 0.045</span>.

המשמעות של הפרמטרים האלה פשוטה:

- <code dir="ltr">max_depth</code> מגביל כמה עמוק העץ יכול לגדול.
- <code dir="ltr">min_samples_split</code> מונע פיצול של קבוצות קטנות מדי.
- <code dir="ltr">min_samples_leaf</code> דואג שלכל עלה תהיה כמות מינימלית של דוגמאות.

### 5.3 מה ראינו

ככל שהעץ עמוק יותר, הוא מתאים טוב יותר לסט האימון, אבל לא תמיד לסט הבדיקה. לכן היה חשוב למצוא עומק שלא יהיה גדול מדי. עבור המודל הטוב ביותר קיבלנו <span dir="ltr">0.867</span> דיוק על סט האימון ו-<span dir="ltr">0.770</span> על סט הבדיקה.

בדקנו גם בחירת תכונות לפי <code dir="ltr">feature_importances_</code>, כלומר אימנו עץ על כל התכונות, דירגנו אותן, ואז ניסינו להשתמש רק ב-<span dir="ltr">top-1</span>, אחר כך <span dir="ltr">top-2</span>, וכן הלאה. המטרה הייתה לבדוק אם אפשר לשפר ביצועים על ידי הסרת מאפיינים פחות חשובים. במקרה הזה, שלוש התכונות המובילות היו <code dir="ltr">cp</code>, <code dir="ltr">ca</code> ו-<code dir="ltr">slope</code>. עם שלושתן קיבלנו <span dir="ltr">CV accuracy = 0.826</span> במקום <span dir="ltr">0.813</span> עם כל 13 התכונות, ודיוק הבדיקה עלה מ-<span dir="ltr">0.770</span> ל-<span dir="ltr">0.787</span>.

---

## 6. מודל שלישי: <span dir="ltr">AdaBoost</span>

### 6.1 איך המודל עובד

<span dir="ltr">AdaBoost</span> הוא מודל אנסמבל שבונה הרבה מסווגים פשוטים מאוד, במקרה שלנו <span dir="ltr">Decision Stumps</span> — עצים בעומק 1.

בכל סבב המודל:

1. מוצא את ה-<span dir="ltr">stump</span> הטוב ביותר.
2. מחשב את המשקל שלו לפי השגיאה.
3. מעלה את המשקל של דוגמאות שסווגו לא נכון.
4. ממשיך לסבב הבא.

המשקל של כל מסווג חושב לפי:

<code dir="ltr">alpha = 0.5 * ln((1-error)/error)</code>

והניבוי הסופי מתקבל משקלול של כל המסווגים יחד:

<code dir="ltr">sign(Σ alpha_i * h_i(x))</code>

כדי להימנע מבעיות מספריות, השתמשנו גם ב-<span dir="ltr">clipping</span> ל-<code dir="ltr">[1e-10, 1-1e-10]</code>.

### 6.2 בחירת מספר הסבבים

בדקנו את הערכים:

<code dir="ltr">[1, 5, 10, 20, 30, 50, 75, 100]</code>

גם כאן נעזרנו ב-<span dir="ltr">5-fold CV</span> כדי לבחור את הערך הטוב ביותר. לפי המחברת, הערך שנבחר היה <span dir="ltr">best n_estimators = 10</span>, עם דיוק <span dir="ltr">CV</span> של <span dir="ltr">0.805</span>.

### 6.3 מה ראינו

ב-<span dir="ltr">AdaBoost</span> לפעמים יותר סבבים באמת משפרים את התוצאה, לפחות עד נקודה מסוימת. לכן הסתכלנו גם על גרף של <span dir="ltr">Train Error</span> מול <span dir="ltr">Test Error</span>, כדי לראות מתי מתחיל <span dir="ltr">overfitting</span>. עבור <span dir="ltr">n_estimators=10</span> קיבלנו דיוק בדיקה של <span dir="ltr">0.787</span>.

בנוסף חישבנו למודל הזה גם <code dir="ltr">feature_importances_</code>, לפי סכום התרומות של כל ה-<span dir="ltr">stumps</span> שהשתמשו באותו מאפיין. גם כאן שלוש התכונות המובילות היו <code dir="ltr">cp</code>, <code dir="ltr">ca</code> ו-<code dir="ltr">slope</code>. ב-<span dir="ltr">CV</span> זה שיפר את התוצאה מ-<span dir="ltr">0.805</span> ל-<span dir="ltr">0.838</span>, אבל על סט הבדיקה לא היה שיפור בפועל: גם עם כל התכונות וגם עם שלוש התכונות הנבחרות התקבל דיוק של <span dir="ltr">0.787</span>.

---

## 7. מודל רביעי: רשת נוירונים

### 7.1 איך המודל עובד

מימשנו <span dir="ltr">MLP</span> מאפס. הרשת כללה:

- שכבות חבויות עם <span dir="ltr">ReLU</span> או <span dir="ltr">tanh</span>
- שכבת פלט עם <span dir="ltr">Sigmoid</span>
- פונקציית <span dir="ltr">Loss</span> מסוג <span dir="ltr">Binary Cross-Entropy</span>
- <span dir="ltr">L2 regularization</span>
- אופטימיזציה עם <span dir="ltr">Mini-batch Gradient Descent</span>

בנוסף השתמשנו באתחול משקולות מתאים: <span dir="ltr">He initialization</span> עבור <span dir="ltr">ReLU</span> ו-<span dir="ltr">Xavier</span> עבור <span dir="ltr">tanh</span>.

### 7.2 בדיקת גרדיאנט

לפני האימון, רצינו לוודא שה-<span dir="ltr">Backpropagation</span> שלנו באמת נכון. לכן עשינו <span dir="ltr">Gradient Check</span> והשווינו בין גרדיאנט אנליטי לגרדיאנט מספרי.

הבדיקה נעשתה לפי:

$$\text{relative diff} = \frac{\|\hat{g} - g\|}{\|\hat{g}\| + \|g\|}$$

כאשר <span dir="ltr">$\hat{g}$</span> הוא הגרדיאנט המספרי ו-<span dir="ltr">$g$</span> הוא הגרדיאנט האנליטי. מבחינתנו, תוצאה של <code dir="ltr">&lt; 1e-5</code> נחשבה הצלחה. בפועל התקבלה תוצאה מצוינת: <code dir="ltr">5.02e-10</code>.

### 7.3 עצירה מוקדמת

כדי למנוע <span dir="ltr">overfitting</span>, הקצנו <span dir="ltr">10%</span> מנתוני האימון כ-<span dir="ltr">validation set</span>. אם ה-<span dir="ltr">validation loss</span> לא השתפר במשך <code dir="ltr">20</code> <span dir="ltr">epochs</span> רצופים, עצרנו את האימון וחזרנו למשקולות הטובות ביותר.

### 7.4 בחירת היפרפרמטרים

הרצנו <span dir="ltr">5-fold CV</span> על כמה ארכיטקטורות וכמה ערכים של קצב למידה ו-<span dir="ltr">L2</span>:

- <code dir="ltr">hidden_layers</code>: <code dir="ltr">(8,), (16,), (32,), (64,), (16,8), (32,16)</code>
- <code dir="ltr">lr</code>: <code dir="ltr">0.1, 0.01</code>
- <code dir="ltr">l2</code>: <code dir="ltr">1e-4, 1e-3, 1e-5</code>

אם היו כמה אפשרויות עם תוצאה דומה, העדפנו רשת קטנה ופשוטה יותר. בפועל, השילוב שקיבל את תוצאת ה-<span dir="ltr">CV</span> הטובה ביותר היה <code dir="ltr">hidden_layers=(64,)</code>, <code dir="ltr">lr=0.1</code>, <code dir="ltr">l2=1e-4</code>, עם <span dir="ltr">0.801 ± 0.051</span>.

### 7.5 מה ראינו

בדקנו גם איך מספר ה-<span dir="ltr">epochs</span> משפיע על המודל, בעזרת הערכים:

<code dir="ltr">[10, 25, 50, 100, 200, 300, 400, 500]</code>

כך היה אפשר לראות מתי המודל עדיין לומד, ומתי הוא כבר מתחיל להתאים את עצמו יותר מדי לנתוני האימון. בריצה הסופית עם <span dir="ltr">Early Stopping</span> האימון נעצר אחרי <span dir="ltr">26</span> אפוקים, כשה-<span dir="ltr">best epoch</span> היה <span dir="ltr">5</span>, ודיוק הבדיקה של המודל הסופי היה <span dir="ltr">0.852</span>.

---

## 8. מודל חמישי: <span dir="ltr">Random Forest</span>

> נוסף לאחר הצגת הפרויקט.

### 8.1 איך המודל עובד

<span dir="ltr">Random Forest</span> הוא מודל אנסמבל שבונה הרבה עצי החלטה ומשלב את התחזיות שלהם. שני מקורות של אקראיות גורמים לעצים להיות שונים זה מזה:

1. כל עץ מאומן על מדגם <span dir="ltr">Bootstrap</span> של השורות (דגימה עם החזרה).
2. בכל פיצול, העץ בוחר את המאפיין הטוב ביותר רק מתוך תת-קבוצה אקראית של מאפיינים (<code dir="ltr">max_features="sqrt"</code>).

בזכות זה העצים פחות מתואמים זה עם זה, והממוצע של התחזיות שלהם מקטין את השונות (<span dir="ltr">Variance</span>) בהשוואה לעץ בודד. כדי לממש את האקראיות הזו הוספנו תמיכה בפרמטר <code dir="ltr">max_features</code> למחלקת עץ ההחלטה הקיימת. המודל מבוסס-עצים, ולכן אינו רגיש לסקאלה של המאפיינים, וניתן לחשב עבורו <code dir="ltr">feature_importances_</code> כממוצע החשיבויות של כל העצים.

### 8.2 בחירת היפרפרמטרים

הרצנו <span dir="ltr">5-fold CV</span> על צירופים של מספר העצים והעומק המרבי:

- <code dir="ltr">n_estimators</code>: <code dir="ltr">[50, 100, 200]</code>
- <code dir="ltr">max_depth</code>: <code dir="ltr">[4, 6, None]</code>

השילוב הטוב ביותר היה <code dir="ltr">n_estimators=50</code> ו-<code dir="ltr">max_depth=6</code>, עם דיוק <span dir="ltr">CV</span> של <span dir="ltr">0.809 ± 0.021</span>. עבור המודל הסופי קיבלנו דיוק אימון של <span dir="ltr">0.975</span> ודיוק בדיקה של <span dir="ltr">0.869</span>.

### 8.3 מה ראינו

בדקנו את גרף ה-<span dir="ltr">Train Error</span> מול ה-<span dir="ltr">Test Error</span> כתלות במספר העצים. בניגוד לעץ בודד, הוספת עצים ליער כמעט אף פעם לא גורמת ל-<span dir="ltr">overfitting</span>: שגיאת הבדיקה ירדה מ-<span dir="ltr">0.213</span> עם עץ אחד ל-<span dir="ltr">0.115</span> עם <span dir="ltr">200</span> עצים ואז התייצבה, בעוד ששגיאת האימון התאפסה במהירות.

בדקנו גם בחירת תכונות לפי <code dir="ltr">feature_importances_</code>. מעניין לראות שביער החשיבות מתפזרת בין יותר מאפיינים (כי כל פיצול רואה רק תת-קבוצה אקראית שלהם), ולכן דירוג המאפיינים שטוח יותר מאשר בעץ בודד. בבחירה הטובה ביותר נבחרו <span dir="ltr">11</span> המאפיינים המובילים, ודיוק הבדיקה עם תת-הקבוצה עלה מ-<span dir="ltr">0.869</span> ל-<span dir="ltr">0.902</span>.

---

## 9. מודל שישי: <span dir="ltr">SVM</span>

> נוסף לאחר הצגת הפרויקט.

### 9.1 איך המודל עובד

מימשנו מאפס <span dir="ltr">Support Vector Machine</span> עם שוליים רכים (<span dir="ltr">Soft-Margin</span>) וגרעין <span dir="ltr">RBF</span>. ה-<span dir="ltr">SVM</span> מחפש את המפריד שממקסם את השוליים בין המחלקות, והגרעין מאפשר הפרדה לא-לינארית על ידי מיפוי מרומז למרחב ממימד גבוה:

$$K(x, z) = \exp\left(-\gamma \, \|x - z\|^2\right)$$

את בעיית האופטימיזציה פתרנו עם גרסה מפושטת של אלגוריתם <span dir="ltr">SMO</span>, שמעדכן בכל צעד זוג מכפילי <span dir="ltr">Lagrange</span> (<code dir="ltr">alpha</code>) עד להתכנסות. כמו <span dir="ltr">k-NN</span> והרשת הנוירונים, זהו מודל מבוסס-מרחק, ולכן חשוב מאוד להזין לו נתונים מנורמלים (<span dir="ltr">z-score</span>).

### 9.2 בחירת היפרפרמטרים

הרצנו <span dir="ltr">5-fold CV</span> על צירופים של פרמטר הרגולריזציה <code dir="ltr">C</code> ופרמטר הגרעין <code dir="ltr">gamma</code>:

- <code dir="ltr">C</code>: <code dir="ltr">[0.01, 0.1, 1, 10, 100]</code>
- <code dir="ltr">gamma</code>: <code dir="ltr">["scale", 0.01, 0.1, 1]</code>

השילוב הטוב ביותר היה <code dir="ltr">C=1</code> ו-<code dir="ltr">gamma=0.1</code>, עם דיוק <span dir="ltr">CV</span> של <span dir="ltr">0.813 ± 0.044</span>. עבור המודל הסופי קיבלנו דיוק אימון של <span dir="ltr">0.934</span> ודיוק בדיקה של <span dir="ltr">0.869</span>.

### 9.3 מה ראינו

ההשפעה של הנרמול הייתה בולטת: בלי נרמול ה-<span dir="ltr">SVM</span> הגיע לדיוק בדיקה של <span dir="ltr">0.639</span> בלבד, ועם נרמול הוא קפץ ל-<span dir="ltr">0.885</span>.

בגרף ה-<span dir="ltr">Train Error</span> מול ה-<span dir="ltr">Test Error</span> כתלות ב-<code dir="ltr">C</code> ראינו את התבנית הקלאסית של <span dir="ltr">overfitting</span>: כאשר <code dir="ltr">C</code> קטן מאוד (<span dir="ltr">0.01</span>) המודל פשוט מדי ונמצא ב-<span dir="ltr">underfitting</span> (שגיאה <span dir="ltr">0.459</span>); סביב <code dir="ltr">C=1</code> מתקבל האיזון הטוב ביותר (שגיאת בדיקה <span dir="ltr">0.131</span>); וכאשר <code dir="ltr">C</code> גדול מדי (<span dir="ltr">100</span>) המודל מתאים את עצמו יתר על המידה לאימון (שגיאת האימון מתאפסת אך שגיאת הבדיקה עולה ל-<span dir="ltr">0.164</span>).

---

## 10. השוואה בין המודלים

אחרי שסיימנו לכוון את כל המודלים, אימנו כל אחד מהם עם ההגדרות הטובות ביותר שמצאנו.

| מודל | נרמול | מאפיינים |
|------|-------|-----------|
| <span dir="ltr">k-NN (k=11)</span> | כן | כל המאפיינים |
| <span dir="ltr">Decision Tree (depth=5, split=15, leaf=5)</span> | לא | <code dir="ltr">cp, ca, slope</code> |
| <span dir="ltr">AdaBoost (n_estimators=10)</span> | לא | <code dir="ltr">cp, ca, slope</code> |
| <span dir="ltr">Random Forest (n_estimators=50, max_depth=6)</span> | לא | <span dir="ltr">11</span> תכונות מובילות |
| <span dir="ltr">SVM (RBF, C=1, gamma=0.1)</span> | כן | כל המאפיינים |
| <span dir="ltr">Neural Network ((64,), lr=0.1, l2=1e-4)</span> | כן | כל המאפיינים |

את התוצאות סיכמנו בצורה הבאה:

<pre dir="ltr"><code>results_df:
               train acc  test acc  overfit gap
Random Forest    0.975      0.902       0.073
k-NN             0.855      0.885      -0.030
SVM              0.934      0.869       0.065
Neural Net       0.842      0.852      -0.010
Decision Tree    0.838      0.787       0.051
AdaBoost         0.838      0.787       0.051</code></pre>

מהטבלה רואים שלאחר הוספת שני המודלים החדשים, המודל הטוב ביותר היה <span dir="ltr">Random Forest</span> עם דיוק בדיקה של <span dir="ltr">0.902</span>. אחריו הגיע <span dir="ltr">k-NN</span> עם <span dir="ltr">0.885</span>, אחריו <span dir="ltr">SVM</span> עם <span dir="ltr">0.869</span>, ואז הרשת הנוירונים עם <span dir="ltr">0.852</span>. עץ ההחלטה ו-<span dir="ltr">AdaBoost</span> נשארו מאחור עם <span dir="ltr">0.787</span> כל אחד. הפער בין <span dir="ltr">train accuracy</span> ל-<span dir="ltr">test accuracy</span> עזר לנו להבין עד כמה כל מודל עושה <span dir="ltr">overfit</span>.

---

## 11. חשיבות המאפיינים

את ניתוח החשיבות עשינו בעזרת עץ ההחלטה, <span dir="ltr">AdaBoost</span> ו-<span dir="ltr">Random Forest</span>, כי בשלושת המודלים האלה אפשר לחשב <code dir="ltr">feature_importances_</code> בצורה ישירה יחסית.

נרמלנו את וקטורי החשיבות כך שהסכום שלהם יהיה <span dir="ltr">1</span>, ואז הסתכלנו גם על הממוצע בין שלושת המודלים.

ראינו שיש כמה מאפיינים שחוזרים שוב ושוב כחשובים. לפי ממוצע החשיבות של שלושת המודלים, שלושת המאפיינים הבולטים ביותר היו <code dir="ltr">cp</code> עם <span dir="ltr">0.221</span>, אחריו <code dir="ltr">ca</code> עם <span dir="ltr">0.144</span>, ואז <code dir="ltr">slope</code> עם <span dir="ltr">0.106</span>. אחריהם הגיעו <code dir="ltr">thalach</code> עם <span dir="ltr">0.084</span> ו-<code dir="ltr">thal</code> עם <span dir="ltr">0.074</span>.

---

## 12. קשיים עיקריים בדרך

### כפילויות במאגר

הכפילויות יצרו תמונה מטעה של ביצועים טובים יותר ממה שבאמת קיבלנו. הפתרון היה להסיר אותן ולעבוד רק עם נתונים נקיים.

### דליפת מידע בנרמול

היה חשוב לא לאמן את ה-<span dir="ltr">Scaler</span> על כל הנתונים יחד. לכן הקפדנו שהוא ילמד רק מ-<code dir="ltr">X_train</code>.

### דליפת מידע ב-<span dir="ltr">Cross Validation</span>

גם בתוך ה-<span dir="ltr">CV</span>, בכל <span dir="ltr">fold</span> אימנו מחדש את ה-<span dir="ltr">Scaler</span> רק על נתוני האימון של אותו סבב.

### מימוש נכון של <span dir="ltr">Backpropagation</span>

כדי להיות בטוחים שהרשת הנוירונים באמת ממומשת נכון, השתמשנו ב-<span dir="ltr">Gradient Check</span>.

### זמן ריצה

במיוחד בעץ ההחלטה, מספר הצירופים שבדקנו היה גדול. למרות זאת, זמן הריצה עדיין היה סביר ואפשר היה לעקוב אחרי ההתקדמות.

---

## 13. סיכום

### האם אפשר לחזות מחלת לב?

כן. כל ששת המודלים הצליחו להגיע לתוצאות טובות יותר מניחוש אקראי, ולכן אפשר לומר שיש במאגר מספיק מידע כדי לבנות מודל חיזוי שימושי. דיוקי הבדיקה במחברת נעו בין <span dir="ltr">0.787</span> ל-<span dir="ltr">0.902</span>.

### איזה מודל היה הכי טוב?

לפי התוצאות הסופיות במחברת, ולאחר הוספת שני המודלים החדשים, המודל הטוב ביותר היה <span dir="ltr">Random Forest</span>, עם דיוק בדיקה של <span dir="ltr">0.902</span>. אחריו הגיע <span dir="ltr">k-NN</span> עם <span dir="ltr">0.885</span>, אחריו <span dir="ltr">SVM</span> עם <span dir="ltr">0.869</span>, ואז הרשת הנוירונים עם <span dir="ltr">0.852</span>, ולבסוף עץ ההחלטה ו-<span dir="ltr">AdaBoost</span> עם <span dir="ltr">0.787</span> כל אחד. מצד שני, עץ החלטה הוא עדיין המודל שהכי קל להסביר ולהבין.

### אילו מאפיינים היו הכי חשובים?

מהניתוח שלנו עלה שהמאפיינים המרכזיים ביותר היו <code dir="ltr">cp</code>, <code dir="ltr">ca</code> ו-<code dir="ltr">slope</code>. אלה גם בדיוק שלוש התכונות שנבחרו גם בעץ ההחלטה וגם ב-<span dir="ltr">AdaBoost</span> בשלב בחירת המאפיינים.

בסך הכול, הפרויקט הראה שלא רק שאפשר לחזות מחלת לב בצורה סבירה, אלא גם שאפשר להבין אילו גורמים משפיעים יותר על התחזית.

---

## 14. קישורים

- קוד: <span dir="ltr">src</span> ו-<span dir="ltr">heart_disease.ipynb</span>
- מאגר הקוד ב-<span dir="ltr">GitHub</span>: <a dir="ltr" href="https://github.com/yonatang8675/ML_Project">github.com/yonatang8675/ML_Project</a>
- מאגר הנתונים: <a dir="ltr" href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">Kaggle - Heart Disease Dataset</a>

</div>
