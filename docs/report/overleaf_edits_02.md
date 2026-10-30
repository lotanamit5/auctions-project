# עריכות לסקשן 2 (המשלימים) בגרסה של השותפה ב-Overleaf

נכתב ב-2026-10-28, עודכן ב-2026-10-29. הבסיס הוא הגרסה של השותפה ב-Overleaf.
ב-repo, הקובץ `02_complementarities.tex` הוא הגרסה שלה, ועוד עריכות 1 עד 3 בלבד.

כל עריכה מסומנת:
- **[כבר ב-repo]**: בוצעה בעותק ב-repo, ונשאר להחיל אותה ב-Overleaf.
- **[לא בוצעה]**: המלצה, שלא בוצעה באף עותק.
- **[דורש החלטה]**: תלוי במה שיש ב-`main.tex` או בבחירה של המחבר. להכריע בסשן הבא.

דרך העבודה המומלצת: להחיל את כל העריכות על העותק ב-repo, ואז להחליף את הקובץ כולו ב-Overleaf.

בגרסה של השותפה הכול נכון מבחינה עובדתית, חוץ מעריכות 1 עד 3. שאר העריכות הן כפילויות ודיוקים קטנים.

---

## חובה: תיקונים עובדתיים

הרקע: הבדיקה של גרף ה-SMD הראתה שהנוסחה הסגורה min{ρ_A, ρ_B} איננה הדוקה עבור ε בין 0 ל-0.24.
ה-LP של התפריטים, שגם הוא חסם עליון, נמוך ממנה עד 0.018, ב-ε≈0.17.

### 1. ניסוח המשפט [כבר ב-repo]

להחליף את:
```latex
and its value has a closed form:
```
ב-:
```latex
and the following theorem bounds it in closed form:
```

### 2. משפט על ההדיקות [כבר ב-repo]

אחרי "...uses only three or four type profiles and no incentive constraint." להוסיף:
```latex
The menu LP, itself an upper bound, coincides with it at $\varepsilon=0$ and for
$\varepsilon\ge0.24$, and is lower by up to $0.018$ in between (at $\varepsilon\approx0.17$, where
$\rho_A$ and $\rho_B$ cross), so there the theorem is not the binding bound.
```

### 3. ה-caption של גרף 03_eps_overview [כבר ב-repo]

בגרסה של השותפה יש שלושה עותקים של ה-caption. רק הראשון פעיל, זה שבתוך ה-minipage, ושני האחרים בהערה. צריך לשנות רק את הפעיל.

להחליף את:
```latex
Dashed: upper bound on fixed-menu
mechanisms (Theorem~\ref{thm:menu_bound}, computed by the menu LP at $N=200$).
```
ב-:
```latex
Dashed: menu LP ($N=200$), an upper bound on fixed-menu
mechanisms that equals Theorem~\ref{thm:menu_bound} at $\varepsilon=0$ and for $\varepsilon\ge0.24$.
```

---

## כפילויות

### 4. מחיקת הבלוק הכפול [כבר ב-repo]

הבלוק מתחיל ב-"This gives the \emph{$\varepsilon$-aware LIP$(a)$}: the menu is the LIP budget set",
ונגמר ב-`\end{itemize}` שלפני `\subsection{Extending the bounds}`. הוא חוזר על טענות 3 עד 5 ועל פסקת ה-certificates. גם הפסקה שפותחת אותו מופיעה פעמיים.

המידע היחיד שנעלם במחיקה הוא "≈0.26ε". לכן קודם צריך להעביר אותו לטענה 4.

להחליף את:
```latex
$\Delta U\leq \frac{\varepsilon Q(1/2)^2}{2}.$
```
ב-:
```latex
$\Delta U\leq \varepsilon Q(\tfrac12)^2/2$ ($\approx0.26\,\varepsilon$ for $a=2$).
```

### 5. סוף פסקת ה-certificates [כבר ב-repo]

להחליף את כל המשפטים מ-"These constructions explain the certification approach;" ועד סוף הפסקה ב-:
```latex
For $\varepsilon$-aware LIP(2) the certified ratios (to $10^{-4}$) at
$\varepsilon=0.01,0.1,0.5,1,2$ are $0.8308,\ 0.8403,\ 0.8100,\ 0.7652,\ 0.7153$.
```

הסיבות:
- המשפט הנוכחי מסתייג מדיוק שכן אושר.
- המספרים אושרו בסבולת 10⁻⁴, במחברת 03, דרך `bracket_lip(2.0, eps, tol=1e-4)`.
- המילה "previously reported" מפנה לטיוטה שהקורא לא מכיר.

---

## דיוקים קטנים בטענות

### 6. טענה 1 (first best) [כבר ב-repo]

להחליף "For reports $x, y \in [0, 1],$" ב-"For types $x, y \in [0, 1],$".
ה-first best מוגדר על הטיפוסים האמיתיים, לא על הדיווחים.

### 7. ההוכחה של טענה 2 (fixed menus) [כבר ב-repo]

ההוכחה הפעילה לא אומרת שהדיווח של מי שאינו הדיקטטור לא משפיע על התוצאה.
ההוכחה שבהערה מתחתיה, שמתחילה ב-"Let $M$ be the dictator's report-independent menu", שלמה יותר.
אפשר להחזיר אותה במקום הפעילה.
בוצע (2026-10-29): ההוכחה מההערה הוחזרה, בתוספת המשפט האחרון של ההוכחה הישנה ("The argument does not require additive valuations..."); ההעתק שבהערה נמחק.

### 8. טענה 4 (manipulability) [כבר ב-repo]

הטענה אומרת "not strategy-proof in general", אבל ההוכחה לא מראה טיפוס שבאמת מרוויח משקר.
המשפט שבהערה אחרי "strictly improves utility" עושה את זה, וכדאי להחזיר אותו:
```latex
For example, for an interior type $v$ whose additive optimizer
$t_0(v)$ is neither $1/2$ nor a boundary point, the derivative of the
complementarity contribution is $\varepsilon P'(t_0(v))$, which is
nonzero whenever $P'(t_0(v))\ne0$.
```
המשפט נכון: P סימטרית וקעורה ממש, ולכן P'(t)=0 רק ב-t=½.
מכאן שכל טיפוס שהבחירה שלו פנימית ושונה מ-½ מרוויח משקר.
בוצע (2026-10-29): המשפט הוחזר, וסופו שונה ל-"which is nonzero because $P$ is symmetric and strictly concave, so $P'(t)=0$ only at $t=1/2$."

### 9. סימון [כבר ב-repo]

בטענה 4 ובהוכחה שלה כתוב `F_v^\epsilon`, ובכל שאר הדוח `\varepsilon`. כדאי לאחד ל-`\varepsilon`.

---

## מבנה

### 10. ההגדרה של Q ו-b [כבר ב-repo]

Q ו-b מוגדרים רק בסקשן 2, בפסקה "From Guo--Conitzer's artificial-currency mechanism to the LIP frontier".
אבל סקשן 1 כבר משתמש בהם, ב-β(a) וב-Lemma 2.
אם אין ב-`main.tex` רקע שמגדיר אותם קודם, צריך להעביר את ההגדרה לסקשן 1 או למבוא.
הוחלט ובוצע (2026-10-29): המבוא (`part_A_summary.tex`, לא עורכים אותו) מגדיר את a ו-b אבל לא את Q.
לכן: (א) בסקשן 1 נוסף "where $Q(t)=\tfrac1a\ln(1+at/b)$ is the share of an item bought with $t$ units of currency";
(ב) בסקשן 2 נמחקה כל הפסקה "From Guo--Conitzer's artificial-currency mechanism to the LIP frontier",
והמשפט שאחרי טענה 2 הורחב ל-"the menu is the paper's LIP frontier $\{(Q(t),Q(1-t)):t\in[0,1]\}$, fixed by the
price schedule and not by the reports".

### 11. הציטוט [כבר ב-repo]

`\citet{GuoConitzer2010}` דורש natbib ורשומה בשם הזה בקובץ ה-bib.
בשאר הדוח כתוב "the paper". כדאי לוודא שהציטוט מתקמפל, או לאחד את הסגנון.
נסגר (2026-10-29): הציטוט נמחק יחד עם הפסקה (עריכה 10). המאמר מצוטט במבוא.

### 12. הנספח של המשלימים, appendix_eps_proofs.tex [כבר ב-repo]

הקובץ הזה שלנו ב-repo, לא של השותפה.
- P1, P2, P3, P5 ו-P6 חוזרים על ההוכחות שבגוף הדוח, ו-P4 חוזר על פסקת ה-certificates.
- P6 אומר שהטענה על ε≥1 נבדקה "numerically". טענה 5 של השותפה מוכיחה אותה אנליטית, כלומר הנספח מתאר תוצאה חלשה מזו שבגוף הדוח.
- כדאי לצמצם את הנספח להוכחת משפט חסם התפריטים ול-P7, ה-LP על המומנטים. לשני אלה בלבד יש הפניות מגוף הדוח.
בוצע (2026-10-30, לניסיון): P1 עד P6 נמחקו; "P7 (moment LP)" נקרא עכשיו "The moment LP", וההפניה בגוף היא
"(Appendix~\ref{app:eps})". המשפט על המונוטוניות של nested LIP (היה ב-P3) נמחק. פתוח: אולי להכניס את ה-moment LP
לגוף, או להחזיר את P1 עד P6 לנספח.

---

## בסיס המודל: פריטים חלוקים (divisible)

הרקע: במאמר הפריטים חלוקים, ו-p_j הוא החלק מפריט j. הערת שוליים 5 במאמר אומרת שאפשר גם לקרוא
את p_j כהסתברות לזכות בפריט בלתי-חלוק. במודל האדיטיבי שתי הקריאות שקולות, כי התועלת לינארית ב-p.
אצלנו, עם ε·z1·z2, הן כבר לא שקולות: תוחלת המכפלה תלויה באיך שתי ההגרלות מצומדות.

המודל שלנו, הדוגמה של הפירמות, ההצעה והקוד כולם בקריאה החלוקה: z הם חלקים, ו-ε-aware LIP נותן
חלקים דטרמיניסטיים, בדיוק כמו LIP במאמר. רק טענה 5 (nested LIP) כתובה בשפה של פריטים
בלתי-חלוקים ("marginal allocation probabilities", "independent item lotteries"). בקריאה החלוקה
nested LIP הוא מנגנון מוגרל שנותן פריטים שלמים, וההשוואה הנכונה היא מול הצרור החלקי
הדטרמיניסטי, לא מול "הגרלות בלתי-תלויות". המספרים לא משתנים: ε-aware LIP נותן m1·m2,
nested נותן min(m1,m2), ה-first best וחסם ה-LP זהים בשתי הקריאות (ה-McCormick envelope הוא
הקמור של גרף z1·z2 על הריבוע, וגם של ארבע הפינות). SMD לא מושפע, כי אצלו תמיד קואורדינטה אחת
שווה 1, ואז המכפלה והמינימום מתלכדים.

הוחלט (2026-10-29): הבסיס החלוק בכל הפרויקט, כמו במאמר. דוגמת הפירמות תקפה בו כמו שהיא
(קיבולות חלוקות, הדוגמה משתמשת רק בהקצאות שלמות, שהן פינות של המודל).

### 13. להצהיר על הבסיס פעם אחת, בפסקת המוטיבציה [כבר ב-repo]

אחרי המשפט שמגדיר את u_v (זה שמסתיים ב-"and $v$ private.") להוסיף:
```latex
As in the paper, the items are divisible and $(z_1,z_2)$ are fractions; mechanisms may
randomise over bundles, and agents are risk neutral.
```

### 14. טענה 5 בשפה של פריטים חלוקים [כבר ב-repo]

להחליף את נוסח הטענה:
```latex
For any fixed marginal allocation probabilities $m_1,m_2$, coupling the
two item allocations through a common uniform random variable weakly
increases the expected complementarity term for \emph{both} agents relative
to independent item lotteries, without changing their additive expected
utilities. If the resulting nested lottery menu is fixed, choosing one's
```
ב-:
```latex
Replace the deterministic fractional bundle $(m_1,m_2)$ by the lottery over whole items
``draw $U\sim\operatorname{Unif}[0,1]$; the dictator wins item $j$ iff $U<m_j$''. The expected
shares, hence both agents' additive utilities, are unchanged, while the expected complementarity
term weakly increases for \emph{both} agents. If the resulting menu of lotteries is fixed, choosing one's
```
ובהוכחה להחליף את:
```latex
Independent item lotteries instead give the respective
probabilities $m_1m_2$ and $(1-m_1)(1-m_2)$.
```
ב-:
```latex
The deterministic fractional bundle instead gives the complementarity
terms $m_1m_2$ and $(1-m_1)(1-m_2)$.
```

בנספח שלנו ב-repo (P6) הניסוח כבר תוקן באותו אופן.

### 15. "lottery" הופך ל-"bundle" בחלק של ε-aware LIP [כבר ב-repo]

בבסיס החלוק ε-aware LIP נותן צרור דטרמיניסטי. המילה "lottery" נשארת רק עבור nested LIP ועבור
המחלקה הכללית. ארבע החלפות:

1. בפסקה שאחרי טענה 2, להחליף:
```latex
and the dictator chooses a lottery maximizing her utility for the reported type and the
public parameter $\varepsilon$.
```
ב-:
```latex
and the dictator chooses a bundle maximizing her utility for the reported type and the
public parameter $\varepsilon$.
```
(סעיף 1 בוצע ב-2026-10-29 במסגרת עריכה 10, במשפט המורחב.)
2. באותה פסקה, להחליף "and the role of lottery correlation." ב-"and the role of randomisation."
3. בניסוח טענה 3, להחליף "Choosing this maximizer from the fixed LIP lottery menu" ב-"Choosing this maximizer from the fixed LIP menu".
4. בהוכחת טענה 3, להחליף "The menu of allocation lotteries is fixed" ב-"The menu of bundles is fixed".

### 16. הגדרת המחלקה לפני משפט חסם התפריטים [לא בוצעה]

nested LIP נמצא מחוץ למרחב התוצאות של המאמר, כי התפריט שלו מכיל הגרלות ולא רק צרורות.
לכן המשפט צריך לומר על איזו מחלקה הוא חל. בתחילת הפסקה
`\paragraph{Upper bound on fixed-menu mechanisms.}`, לפני "The best fixed-menu mechanism on a grid",
להוסיף:
```latex
A \emph{fixed-menu mechanism} lets a fair coin pick a dictator who chooses from a fixed menu of
lotteries over bundles; the class contains the paper's swap-dictatorial mechanisms (deterministic
menus) and both LIP variants above.
```
ובמשפט עצמו להחליף "by the two-item menu" ב-"by the deterministic two-item menu".

---

## מה נבדק ונמצא תקין בגרסה של השותפה

- **דוגמת הפירמות:** המספרים 1.8, 1.5 ו-1.
- **ההגדרה של Q ושל b:** כולל Q(0)=0 ו-Q(1)=1.
- **ההוכחה של טענה 3:** קעירות ממש, הפרשים עולים ו-Topkis.
- **החסם בטענה 4:** εQ(½)²/2, דרך P(t)≤P(½), וחצי מהגרלת הדיקטטור.
- **ההוכחה של טענה 5:** אי-השוויונות על G'(t) בשני הצדדים של ½, עבור ε≥1.
- **מספרים:** 0.004 ו-0.003 בסיכום המקרה האדיטיבי, ו-0.857 מול 0.951 ב"מחיר הדיקטטורה".
- **הדיקטטור הצנוע:** s\*<½.
