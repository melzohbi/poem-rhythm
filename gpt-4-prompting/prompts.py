import tiktoken

# to calculate the number of tokens in the prompt
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# these are the instructions for the task
instructions = """
You are given a dataset of poetry verses where certain words have been masked with `<E0>` and `<E1>` tokens. These tokens represent the start and end of a masked word or phrase in the verse. Each masked word has a "beat pattern" defined by a sequence of 1's and 0's, where:
- 1 indicates a vowel onset (transition from a consonant to a vowel),
- 0 indicates no vowel onset.

Your task is to reconstruct the masked words or phrases based on the provided beat patterns. To help you understand the structure, here are some examples of words with their consonant-vowel patterns and corresponding beat patterns:
- "Woods" has a consonant-vowel pattern of `CVCC` and a beat pattern of `100`.
- "Spirit" has a consonant-vowel pattern of `CCVCVC`, corresponding to a beat pattern of `0110`.
- "The nice day" has a consonant-vowel pattern of `CVCVVCCVV`, resulting in a beat pattern of `110010`.
- "I believe in" has a consonant-vowel pattern of `CVVCVCVVCVC`, creating a beat pattern of `1011010`.

Below are examples of poetry verses with masked words. Each masked word is represented by a beat pattern. Your job is to fill in the correct word or phrase to replace the `<E0>` and `<E1>` tokens based on the provided beat pattern and the context of the verse.

Verses:
1 <E0>110<E1> torments me from his western bed
2 <E0>1010110<E1> of earth to earth

Expected reconstruction:
1, The sun, CVCVC, 110
2, Indignity, CVCCVCCVCVV, 1010110

Given this information, reconstruct the masked words in the following verses based on their beat patterns and context. Return: the example number, the expected masked word, the CV pattern and the beat pattern ONLY.

"""

# here is the prompt for the task including the instructions and the first 250 examples
prompt_text_250 = instructions + """
1	Through length <E0>10<E1> years of life beside
2	Shall hang them up in silent <E0>101100<E1>
3	And after April when <E0>10<E1> follows
4	To dream to night on you my <E0>10<E1>
5	After <E0>1<E1> billows of tempestuous oceans
6	That ran <E0>10<E1> duty's call
7	But not upon my <E0>1000<E1>
8	To mark for <E0>10<E1> than fourscore years a line
9	The final splendid <E0>100100<E1> of his script
10	Before the adorable god <E0>1<E1> infant Jesus
11	The pilgrim <E0>10<E1> his thirst assuage
12	<E0>1010<E1> the silver cornets clear and high
13	Something heard I like the <E0>001010<E1>
14	<E0>1<E1> youth exerted ev'ry art to please
15	Strange rendezvous My mind was <E0>1010<E1> time
16	<E0>10<E1> its end could never be found
17	These <E0>010010<E1> had the power to speak
18	Heed those old <E0>01100<E1>
19	The sunset <E0>0101<E1> died away
20	The blanket stiff <E0>10<E1> packs his bed
21	<E0>100<E1> Once the Galatians built a fane
22	The <E0>101<E1> sells his other merchandize
23	<E0>10<E1> least it seems so for they all act suited
24	In silence I <E0>0100<E1>
25	Where howled the wolf and ached <E0>11010<E1> plain
26	Of Jesus <E0>10<E1> this sinful earth
27	Palace farm villa shop <E0>100<E1> banking house
28	Piping on boughs or sporting <E0>100010<E1> fields
29	I'll comfort thee my dearie <E0>10<E1>
30	Baring <E0>1110<E1> on which sweet Sydneys name
31	The stables <E0>10<E1> alive with din
32	<E0>100<E1> Argo saw her kindred trees
33	What consummation <E0>1000<E1> apace
34	Sure <E0>1010<E1> game But most the wisdom shows
35	Their phalanx to replenish <E0>1010<E1> driven
36	And he bayed to the <E0>10010<E1> she rose
37	Rise to that home all <E0>10000<E1> above
38	The late light falls <E0>1010<E1> the floor
39	Oft as he sang that hand lovely <E0>10<E1> light
40	Standing <E0>1001<E1> than the door
41	How their sweetest smile is kept for <E0>10<E1>
42	Where I left you on <E0>1<E1> plain
43	Tones of more godlike <E0>01000<E1>
44	Fast fly the hours and all will soon <E0>1010<E1>
45	For the <E0>101100<E1> the wormwood of the whole
46	<E0>100<E1> the soul forever liveth
47	The human hurricane <E0>101000<E1> can see
48	Where by the winding Ayr <E0>10<E1> met
49	Break open to their highest <E0>10010<E1> the stars
50	Among them he a spirit of <E0>0101<E1> sent
51	Who shall say it Who may <E0>10<E1> it
52	Honor the charge <E0>10<E1> made
53	To heal <E0>100<E1> love and bless
54	Deathly cold <E0>0100<E1> up the tide
55	<E0>10000<E1> of the street
56	As <E0>1010<E1> rare fair woman I am now but a thought of hers
57	In the fields of your <E0>110100<E1>
58	Much too <E0>10<E1> to touch at all
59	And ruled o'er Denmark <E0>1001<E1> heathy isles
60	Rings through their lowings dull though heard by <E0>010<E1>
61	In loftier <E0>1000<E1> defied the simple bird
62	But as I now believ'd <E0>10<E1> dead
63	Philosophy <E0>10100<E1> on Grecian eyes
64	Where <E0>10100<E1> crawled they chawed all greening things
65	<E0>100<E1> lay their work away
66	Is it music from <E0>1<E1> graves
67	Yet the old canoe is safely on the shore where you can let <E0>10<E1>
68	Oh had we some <E0>0100110<E1> isle of our own
69	Shall the <E0>0100<E1> wisdom of our patriot sires
70	And fed by rapine sleep soft <E0>100<E1> away
71	God <E0>1<E1> Father we will adore
72	Nor a <E0>100<E1> that's niver cast
73	The <E0>10010<E1> blood is at your door
74	<E0>10<E1> the days are growing longer
75	Onaway my heart sings to <E0>10<E1>
76	Are weakened in our <E0>0100<E1> in Thee
77	From A Jovial Crew by RICHARD BROME Enter Patrico <E0>1010<E1>
78	Oh sweet <E0>1<E1> life of a Road
79	Leaving it blind of a <E0>1010<E1> star
80	That <E0>100<E1> wad a haen him
81	A little farther <E0>101010<E1> they wish to hear the truth
82	<E0>10<E1> we greet thee rare sweet maiden
83	When <E0>101<E1> the bracken on the grass
84	GREAT <E0>01000<E1> was born
85	Conscience and Fame <E0>10<E1> sacrificed to Rome
86	Mary <E0>0100<E1> married Webster
87	And when <E0>10<E1> sisters birthdays came
88	Their offals or their better <E0>100<E1>
89	<E0>10<E1> under leaden hail with flaming breath
90	Each season did its <E0>010100<E1> bring
91	Blown back <E0>100<E1> every wind
92	Will go with <E0>10<E1> along
93	Come draw a drap o <E0>1<E1> best o't yet
94	Might <E0>10<E1> to serious thoughts some wealthy men
95	Then Maeve <E0>101010<E1> Master of all lovers
96	Yet Teviot's sons with <E0>10<E1> disdain
97	Where I wandered all <E0>100<E1> night
98	To the lumberer asleep <E0>100<E1> thy glooming
99	Sometimes the spirit that <E0>111000<E1> me quite
100	I know he loves the land his <E0>010<E1> has won
101	By <E0>110<E1> me away
102	Now with fine fingers <E0>1010<E1> cold
103	You've brought us canned beef <E0>10100<E1> now my belief
104	<E0>10<E1> is but one whiche'er it be
105	<E0>10<E1> bring enchantment so profuse
106	Us higher not through craven <E0>100<E1>
107	<E0>10100<E1> the Delphic priestess and the pale
108	Not thou had been to <E0>0100<E1>
109	The wild deer browse above <E0>10<E1> breast
110	Had flashed its whiteness <E0>010<E1> afar
111	Suddenly in my memory <E0>100<E1>
112	The <E0>10110<E1> cold and I have no vest
113	The shining mists are <E0>10<E1>
114	The liberated captives <E0>100<E1> applause
115	And all the wings of the <E0>100100<E1> all the joy before death
116	How <E0>100<E1> law to bear eluding Not because of impotence
117	<E0>1010<E1> Zeus relenting the mandate was revoked
118	<E0>10<E1> Ftihah forth beneath the Mehrab board
119	That day the daisy had an eye <E0>10100<E1>
120	<E0>1010<E1> not bear earth s overwhelming strife
121	The <E0>101<E1> honey sucked from myriad flowers
122	O little master <E0>100<E1> to me
123	How good it would be my <E0>011<E1>
124	<E0>100<E1> wonder not for when a generall losse
125	Thy flitting form comes ghostly dim <E0>100100<E1>
126	Sparkling so <E0>0100<E1> upon the stream
127	The <E0>10<E1> and heat whose bubbles make
128	Why aught should fail and fade that once <E0>10100<E1>
129	I <E0>100001<E1> slow withdrawal of the year
130	While <E0>100<E1> caught fast in pleasure's chain
131	But <E0>10100<E1> easy with a mind like ours
132	Death's shafts fly thick and love a <E0>10101000<E1>
133	Now the plantin taps are tinged <E0>10100100<E1> yon burn side
134	Nor <E0>101<E1> nor example with him wrought
135	Wide love for all it is <E0>101<E1> worthless thing
136	Your <E0>110<E1> dog who still lives near
137	When good nights have been prattled and prayers have been <E0>10<E1>
138	Again <E0>10100<E1> be great Quick come the hour
139	And weep <E0>100100<E1> thyself thy anger vent
140	To point a moral and <E0>11001<E1> fable
141	<E0>10010<E1> the sparkling waves in glee
142	And how the steadier will <E0>10100<E1> succeed
143	Of old <E0>1<E1> fire tongued miracle
144	And <E0>10<E1> thy own sermon thou
145	Even as I see and share <E0>10<E1> you in seeing
146	<E0>10<E1> he humble or great
147	Redeemed from sickness death <E0>100<E1> pain
148	Ah <E0>10<E1> delight wilt thou receive me
149	He praised <E0>1010<E1> the glory of thy sex
150	There was this for us to say and there was this for <E0>101010<E1>
151	A warlike chieftain was their <E0>100<E1> request
152	<E0>10<E1> hurls a heavy raft along
153	And then again when it grows <E0>100<E1>
154	Hark hark <E0>1<E1> shepherd's voice Oh sweet
155	The <E0>10100<E1> in his wisdom had them so
156	To <E0>100<E1> their wings to the wide wild air
157	<E0>10001000<E1> and mist starriness over the trees
158	That fills <E0>10010010<E1> with the idle words
159	But weel the <E0>10010<E1> lover marks
160	O <E0>10<E1> its memory like a chain about thee
161	Then I'll <E0>10010<E1> its glory and rest till I see
162	<E0>100<E1> the sunbeams shone the zephyrs played
163	A tattoo on the <E0>100<E1>
164	The gray gulls are flying <E0>10010<E1> sail swings
165	<E0>10010<E1> shall conquer who thinks he can
166	While lustrums <E0>100<E1> their way
167	<E0>10<E1> housewife worthy of a morning visit
168	But not tree <E0>11010<E1> from her planting
169	Earth has no sorrow that <E0>1101100<E1> cure
170	<E0>10<E1> came as with a gently swelling tide
171	<E0>100100<E1> ye home and with you bear
172	Why tear their flesh in Corpes wood with saddle girths and <E0>0100<E1>
173	Those silent counsellors <E0>10<E1> ladies prize
174	Now <E0>10010<E1> heads together at your will
175	Its <E0>1010<E1> is not spent
176	Within the hollow <E0>10<E1> His mighty hand
177	Never the second time <E0>1010<E1> me tell you a story
178	That grow about <E0>1<E1> beds and bowers
179	Above <E0>1<E1> crystalline height
180	For a bad <E0>10000<E1> punishment
181	Another <E0>10<E1> the nuptial name
182	And love you know hath <E0>01000<E1> feet
183	<E0>1010000<E1> my annual verse to pay
184	Then was <E0>1<E1> lyre of earth beheld
185	Slave to the vices that she <E0>11100<E1>
186	And summon'd to partake its <E0>1100<E1> woe
187	When one <E0>11010<E1> ghost
188	<E0>1<E1> horse is to a T
189	And yearned to be <E0>10<E1> home
190	But <E0>1010<E1> has to earn his bread
191	And the white breast of the <E0>1010<E1>
192	The <E0>101<E1> love of a mother's hands
193	Long ages ere that <E0>10<E1>
194	And all around the winds are <E0>0100<E1>
195	Hang curtains round <E0>1<E1> day
196	Beneath its <E0>00100<E1> of scented leaves
197	<E0>10<E1> was once a little kite
198	And just because I was thrice <E0>10<E1> old
199	<E0>10<E1> Osseo the Magician
200	Till he found his furlough strangely <E0>100010<E1> win
201	<E0>10100100<E1> fallen from heaven
202	Attend O Pallas <E0>100<E1> with lifted arm
203	By <E0>1<E1> cares of yesterday
204	And pledge the gath'ring of <E0>1<E1> host
205	Nor less Winander <E0>1010<E1> power I owe
206	To dreams and shapes of shadowy things <E0>11000<E1>
207	Then awake the heavens look bright my <E0>10<E1>
208	Haste with my <E0>1000<E1> my incenses and myrrhs
209	Here cool fresh spirits the air <E0>010<E1>
210	The <E0>110<E1> shames and scorns
211	On any pretext <E0>10100<E1>
212	<E0>10<E1> tackle the country dorgs comin to town
213	Ah me the blooming pride <E0>10<E1> May
214	And that it meant <E0>110010<E1> him when I'd done something fine
215	Softly <E0>1<E1> thousand shattered men
216	Rest rest he'll <E0>100<E1> be here
217	And <E0>10<E1> contentment too
218	Never at all since I <E0>100<E1> here
219	Down the old <E0>10<E1> winding yonder
220	Tilt stagger <E0>100<E1> plunge over
221	Yet when we've toiled <E0>1011<E1> a weary day
222	With islands where a Goddess walks <E0>1100<E1>
223	By <E0>100101000<E1> may point her sharpest strain
224	Their sound is but their stir they <E0>010010<E1> silences
225	And death or restitution is <E0>1<E1> word
226	With swollen cheeks clangs <E0>10<E1> alarms
227	Could fear <E0>10<E1> with a smile
228	The banns were read <E0>100100<E1> was sewn
229	<E0>10<E1> leaves o'er the bed
230	New and <E0>10<E1> and thrice so new
231	No times past present <E0>1010<E1> come could e'er
232	Howe'r <E0>10<E1> trust to mortal things
233	With rays of pardon <E0>10<E1> the World's offense
234	But I did not care <E0>1010<E1> hastened by
235	For <E0>100<E1> a little boy he seemed
236	On that blest <E0>10<E1> of Mary Magdalen
237	A sum of hidden <E0>1000<E1>
238	And touched my brown beard with their <E0>101<E1> wands
239	Its empires may <E0>1010<E1> and fall
240	The chief <E0>10<E1> mandarins the great Goh Bang
241	<E0>10<E1> will it serve at last
242	<E0>100<E1> that your nose
243	<E0>100<E1> the bold stars does hear
244	And sends <E0>1010<E1> thy cabin well prepared
245	<E0>010<E1> make the past with all its sin
246	Wildered by the <E0>1101010<E1> of the main
247	To <E0>10<E1> the poorest room
248	A thousand hearts in the crowd and the even <E0>1101010<E1>
249	Wishin fer you <E0>10100<E1> when
250	A vague and <E0>01010<E1> magic
"""

prompt_text_500 = instructions + """
251	And <E0>11000<E1> thy cordial smile
252	Where <E0>10<E1> the hollow arch of space
253	As to stray like mine <E0>10<E1> bliss
254	Ourselves inveterate rogues <E0>10<E1> be
255	Ere the sun <E0>010110<E1> once more has rolled
256	<E0>110<E1> and well a day
257	<E0>10<E1> lady my lady
258	<E0>1010<E1> make up fresh adventures for the morrow do you say
259	There's <E0>1<E1> sunny Southern land
260	<E0>10<E1> troops of angels trying
261	The <E0>1000110100<E1> your great heart the sun
262	And withered in the <E0>10100<E1> lie
263	Slenderly busy <E0>010100<E1> too
264	Little we <E0>100010<E1> an oft turned sod
265	With <E0>100<E1> and wings
266	And oft repell'd th invader's <E0>100<E1>
267	and don't be making a <E0>10010<E1> the street
268	<E0>1010<E1> the heaps of slain he lie
269	Until the stale material will <E0>1100<E1>
270	New <E0>101010<E1> in white and green
271	<E0>10010<E1> the creaking wagons strain
272	Down the <E0>10010<E1> through the glen
273	<E0>10<E1> meet again beyond the barren past
274	<E0>101000<E1> know what in the world it means
275	At <E0>010000<E1> at Batson's and at Will's
276	To <E0>010101010<E1> through the world's wide waste
277	Of Orpheus whom the streams stood <E0>01010<E1> hear
278	Then <E0>10<E1> it see a meeting stern
279	<E0>1<E1> teeming ripple of Boebeis lapped
280	Blazing and cracking away due honour to pay to the <E0>100100<E1>
281	A strong authority which <E0>100<E1> convince
282	<E0>1<E1> multitude of days still heaped on
283	The feast of <E0>101001010<E1> and the Cereal Games IZ
284	This slave of <E0>01010<E1> for the sake
285	<E0>10<E1> bury our dead
286	As <E0>1010<E1> long has lain in wait
287	My tailor serves you well from a perger to a <E0>010<E1>
288	In the sunset's <E0>01010<E1> glow
289	I shall never fight a battle higher up than eagles <E0>10<E1>
290	Upon them <E0>10<E1> is seen his name
291	The love and honor <E0>10<E1>
292	This King he said from <E0>10010<E1> old
293	<E0>10<E1> Briggs Base Eight American
294	<E0>10<E1> you not also pity those
295	On slender stems the <E0>1010<E1> wind flowers blow
296	This excellent letter <E0>101000010<E1> Stewart of Dalguise is copied
297	When a brewery looms on the left <E0>10100<E1>
298	Droop and <E0>11<E1> one by one
299	He mourns that day so soon has glided <E0>10<E1>
300	Close to <E0>10100<E1> oh may I stay
301	At fifty chides <E0>10<E1> infamous delay
302	And lest <E0>110001<E1> throng that make us groan
303	Come over <E0>10<E1> over the deepening river
304	Oft when <E0>10<E1> hear his waltzes sweet
305	All brave in arms <E0>10<E1> trained to wield
306	With a love that was such as <E0>10<E1>
307	No readers here <E0>101010<E1> looks are found
308	Inspiring <E0>010<E1> by turns a lump
309	Can yet be <E0>1010<E1> claim a tear
310	Come let me lead thee o'er <E0>101100<E1> Rome
311	The th verse of the th chapter <E0>1010110<E1> the longest
312	White foxes <E0>010<E1> with seeming innocence
313	While on his lips an expression still <E0>1010<E1>
314	I ought to admire much more not <E0>10<E1>
315	Like Henry VIII in England and Louis XI in France James <E0>10<E1>
316	By certain proofs not few intrepid <E0>1000<E1>
317	Nor doth the Soueraigne of <E0>10100<E1> golden fires
318	Where are you you <E0>10<E1> you
319	Those words again I love you love you <E0>0100<E1>
320	The cobwebb'd cottage <E0>10<E1> its ragged wall
321	Meeting all <E0>1<E1> needs of living
322	To own it touch it <E0>01001<E1> feat
323	To give the <E0>101100<E1> the great alarms
324	<E0>1<E1> truth that has long lain buried
325	A score in all to watch the river <E0>11<E1>
326	If <E0>10<E1> your real substance move
327	Into <E0>10<E1> solitary walk
328	Lulled the <E0>0101<E1> landscape basks
329	Could venture for the golden <E0>0100<E1>
330	So that it seemed her soul was <E0>101<E1> fleet
331	Tender as dawn's first hill fire and <E0>10100<E1>
332	True Friendship <E0>0100<E1> the ivy green
333	Over thee <E0>10<E1> Gismond's dust
334	Where wine <E0>110<E1> a goodly crowd
335	Speak them by Him O <E0>101<E1> unaware
336	The shadow by my finger <E0>100<E1>
337	The cursed race upon <E0>10<E1> fell
338	I am a hunter in <E0>11100<E1> jungle
339	And words <E0>101000<E1> for certain marks to be
340	And Mercy cried <E0>100010<E1> I tell the truth
341	And <E0>1010<E1> waste the lonely hours
342	Of king <E0>0100110<E1> bishop knight and pawn
343	Ay <E0>10010<E1> seemed some strange delight to find
344	Her panting life and <E0>101010<E1> it seemed
345	<E0>100<E1> grew beneath his gaze
346	And <E0>110100<E1> a low voice sweetly humming
347	Sir <E0>0100<E1> the knight of the Fen
348	Down to the past he bears the <E0>100<E1>
349	And gentlemen took supper <E0>10<E1> the street
350	<E0>1<E1> daffodil is our doorside queen
351	And borne <E0>1010100<E1> tribes slavery and pangs
352	For in it she <E0>10100<E1> such pride
353	<E0>10010<E1> some BRANCHES of the selfe same kinde
354	KARSHISH THE ARAB <E0>1110<E1>
355	No <E0>10<E1> that falls beneath it
356	For <E0>100<E1> nae langer must I stay
357	But to <E0>1010100110<E1> haste O Raghu's son lest in his ire
358	I would not see them <E0>10<E1> mine eyes
359	Swift had the sin of wit no venial <E0>0100<E1>
360	And fame's unfadin <E0>01000<E1>
361	<E0>100<E1> leave the rest
362	<E0>1000<E1> had nested in his hair
363	In love with <E0>100100<E1> glad and good
364	Nicholas Rowe Some Account of the Life of Mr <E0>1010<E1>
365	Whose <E0>01000100<E1> like a madman's I am young
366	Or <E0>100<E1> or row
367	Proud <E0>1000<E1> you cannot banish us
368	He gave the mantling vine to <E0>010<E1>
369	Chin dimpling <E0>11<E1> good to wear
370	Theirs is the <E0>10010<E1> mind
371	And back <E0>11010<E1> to your firelit friends
372	To a pond <E0>10<E1> quiet water
373	Thy <E0>0100<E1> prophetic strain we hear
374	And <E0>1010<E1> touch me as He walks serene
375	Where all <E0>1<E1> children dine at five
376	<E0>1<E1> lights of sunset gleam
377	<E0>10<E1> welcome the Traveling Man
378	<E0>10<E1> bound away for ever
379	The group <E0>10011<E1> gay
380	<E0>101000<E1> only breathes immortal air
381	Keep our hearts in <E0>10<E1> safe keeping
382	Or how <E0>1100<E1> the rout and horrid roar
383	Let cankering moss obscure the <E0>100100<E1> name
384	From the full resource of <E0>101010<E1> dome
385	O'er the three worlds his empire <E0>100<E1>
386	Over seas <E0>10<E1> wreck and drown
387	Yet gives no vails <E0>10100<E1> him thence
388	The Great <E0>0110<E1> the Creator
389	This <E0>1<E1> first day of wonders
390	Twilight <E0>10<E1> veiled the little flower face
391	He said Though thou art <E0>10010<E1> great
392	Yet <E0>1<E1> dark Angel's touch
393	Are the <E0>01000<E1> more sweet than mine
394	Unfix Benledi <E0>010<E1> his stance
395	Thou art not <E0>10100<E1> dark mysterious Night
396	No let the rest for <E0>1001000100<E1> accede to slavish terms
397	And my <E0>100<E1> comes back to me
398	It was <E0>1<E1> black and cindered thing
399	I <E0>10<E1> be nearer understood
400	Here are <E0>10<E1> unhorizoned skies
401	Of the <E0>010010<E1> Gudrun the Daughter of Giuki
402	<E0>110<E1> to Doctor Doode Doo
403	Triumphs and well I remember a story that often <E0>101000<E1> me
404	A priest is call'd tis now alas <E0>10100<E1>
405	<E0>10<E1> more to roam
406	And floppy ears to <E0>10<E1>
407	The mossy roofs adore thou <E0>11<E1> sun
408	<E0>101000<E1> could pipe in skies so dull and gray
409	The man who hopes t obtain the <E0>01010010<E1>
410	With <E0>10<E1> scarred backs and labour broken knees
411	Had got <E0>1<E1> pair of chimney corners
412	From Heaven's high places <E0>101100<E1> a king
413	And cheer our <E0>1010<E1> round
414	Wilt thou trust <E0>1010<E1> not He answered Yes
415	Our shallop <E0>10<E1> but slight
416	The King's Royal <E0>0101100<E1> hundred a year
417	And had filled every one <E0>1010<E1> gibbous crops
418	But all their love is <E0>110<E1>
419	The rugged trunk indented deep with <E0>01000<E1>
420	Let any boy in school taste but when <E0>10<E1>
421	With a <E0>110<E1> caravan
422	And realms <E0>1010<E1> dissolved and empires be no more
423	Scenes <E0>101000<E1> Tiber for the mighty dead
424	Far from their native sun <E0>100<E1> shade
425	Nor took his pleasure <E0>10<E1> it
426	Far back <E0>10<E1> youth's valley of hope
427	I <E0>1100100<E1> that thou shouldst pass away
428	Love never in <E0>10<E1> spirit glows
429	That bane <E0>10<E1> our romantic triflers shun
430	Who loved me for <E0>10100<E1> alone
431	What I <E0>100<E1> introduce her
432	In a <E0>10<E1> silvery sheet
433	And stricken her poor father <E0>100<E1>
434	Blush <E0>10<E1> as roses bursting into bloom
435	And Poets are <E0>10<E1> thick as Peas
436	For <E0>10<E1> at last to capture
437	Sad days when <E0>1<E1> sun
438	No more will the Napa catch our <E0>10<E1>
439	<E0>1000<E1> down the west for me
440	But still the shrubs <E0>1010<E1> admires dispense
441	And did I hear or dream them all <E0>10100<E1>
442	Well tim'd now the frost is <E0>10<E1> in
443	<E0>10<E1> the blue vault of the sky
444	<E0>110<E1> in front of them
445	In <E0>10100<E1> nations spake his tongue might be
446	A line in commendation of <E0>10<E1> friend
447	She was my dear and nearest friend to love and pity <E0>10<E1>
448	They'd <E0>0100<E1> the expense to ashume
449	Do as you please your <E0>1010<E1> mine
450	But <E0>010<E1> the throng one dervish pressed
451	From dawn till set of <E0>10<E1> and then
452	<E0>100<E1> blood and mind in freest boon
453	That soul <E0>10<E1> sensibly sedate
454	East and West <E0>010<E1> Dee to Yare
455	On which invention shall be long <E0>100100<E1>
456	The sufferings of <E0>1<E1> sin sick earth
457	Waiting his master <E0>10<E1> will stand
458	Their land is in a <E0>0100<E1>
459	My outward frame doth <E0>10<E1> me this
460	<E0>101<E1> halting step of his age outran
461	Tis a <E0>10110<E1> and timorous beauty
462	Than I <E0>10<E1> put a span of horses in
463	With pains and joys each morning <E0>010<E1>
464	Sweet beam'd the star of peace upon <E0>100100<E1>
465	What nobler <E0>101100<E1> can deck our isle
466	With Plant Fruit Flour <E0>1001010<E1> Gemms Gold
467	As the day to a close <E0>11101010<E1> ran
468	Equal for none <E0>100<E1> great antagonists
469	Smiling as <E0>10<E1> thought of sipping
470	Made a woman slim and <E0>10<E1>
471	That happy <E0>1101<E1> night that night so dark and shady
472	Performed on <E0>10010<E1> pyres
473	Whispering that name of Wife I <E0>100<E1>
474	The song of <E0>1100<E1> to the bee
475	O maiden fair O maiden fair <E0>1010010<E1> is thy bosom
476	With <E0>10<E1> much midnight oil destroyed
477	My strength and brace my <E0>100010<E1> do and dare
478	Shook <E0>010010<E1> of sweet indecisive sound
479	Where sparkles <E0>10<E1> golden splendor
480	<E0>11000<E1> man offer him a place or offer
481	They faint they <E0>011<E1> to and fro
482	Humbly we gaze <E0>100<E1> the colossal frame
483	Or dream <E0>1<E1> winter out in caves below
484	At your side that eve <E0>10<E1> should not have seen
485	<E0>1001<E1> silence and stars and her lips again
486	<E0>10<E1> shall know not a word of their fate
487	Smacks of the merry men He's <E0>011010<E1>
488	Light breaks flowerwise into <E0>10<E1> born sight
489	Sick to thy soul of party noise <E0>10000100<E1>
490	Dropt <E0>100<E1> in danger from passing feet
491	Beckon no more shades of the noble <E0>10<E1>
492	Fanning the busy dreams from <E0>1010<E1> eyes
493	<E0>100<E1> our best reason darken'd his despair
494	Divine the sov'ran Architect <E0>10<E1> fram'd
495	Saw <E0>10<E1> at last
496	<E0>101000<E1> give new things away
497	While everything is still as nature's <E0>10<E1>
498	Wonder d again and look d upon <E0>10100<E1>
499	Mysterious in the <E0>01001<E1> fancying
500	But to my <E0>011100<E1> a well fed Rat
"""


prompt_text_750 = instructions + """
501	Behold <E0>1<E1> multitudes that come and go
502	Across the pool their <E0>10000<E1> unite
503	Rain <E0>101<E1> glad refresher of the grain
504	Oh strong are <E0>1010000<E1> and sweet my rest
505	Fair fruits <E0>10<E1> in the hand
506	The lofty mountains <E0>100<E1> the deeper glade
507	But daily <E0>11<E1> men draw near
508	<E0>101010<E1> want I comfort those who do
509	As faint as lilies on a <E0>1001<E1> noon
510	There are only trees now <E0>100100<E1> no eighties
511	<E0>101000<E1> the center fielder's garb the Mudvilles shirt of red
512	Thy <E0>010010<E1> white thy form too slender
513	From low bough <E0>10<E1> bough
514	Nor joy <E0>10<E1> sorrow
515	<E0>10<E1> which Time himself devours
516	The Earth to yeild unsavourie food <E0>1100<E1>
517	Answered That <E0>100<E1> Norway breaking
518	And the whole <E0>10<E1> land is a red red hell
519	<E0>10<E1> she refrain from the fields
520	Of the world's praise from dark <E0>101011<E1>
521	Against the morning sun <E0>1010<E1> that look
522	In the midst of <E0>1<E1> surging crowd
523	Who cannot bear to <E0>100<E1> back to the departure
524	Still darkly struggles waked from <E0>1010<E1> fear
525	Eight mighty <E0>110000<E1> nosing out their trail
526	Convert you into stores to <E0>010<E1> in rents
527	And vain deceit <E0>10100<E1> where is the truth
528	Of <E0>1<E1> great tree electorate
529	I lived in quiet <E0>1000<E1> content
530	Until he reached <E0>10<E1> father's door
531	Of <E0>10100<E1> moon that a good hour
532	The Count Rollnd feels now <E0>10100<E1> approach
533	A few good old books which <E0>10100<E1> one has read
534	Shrink to the measure of <E0>1<E1> grave
535	For lust and <E0>1010<E1> on he'll go
536	You <E0>11<E1> are a bird of spirit
537	<E0>10010010<E1> her increase Thy rams are there
538	<E0>10<E1> the narrow lanes of life
539	With jingling <E0>100<E1> about her neck
540	With tangled boughs I wander <E0>10<E1> alone
541	Its wide doors <E0>10<E1> the sun
542	<E0>1<E1> strength we need
543	And <E0>10<E1> a poodle ran away
544	<E0>100<E1> ever and ever thereafter with thee
545	As red as any <E0>010<E1>
546	He questions the <E0>1010100<E1> work of man
547	<E0>100010<E1> it be would charm your ear
548	A grace not <E0>1001<E1> but more rare
549	<E0>11000<E1> him much preparing
550	Too blind too blind to <E0>10<E1>
551	A <E0>010110<E1> captive all aflame
552	A rival of its <E0>110<E1>
553	Vainly the <E0>010100<E1> of Baal would rend it
554	And quivering <E0>1001<E1> to the roving breeze
555	Its sacredness to <E0>100010100<E1>
556	Your <E0>100<E1> that were
557	<E0>10<E1> the stream that hems my path
558	The deep came up with its chanting <E0>1000<E1>
559	His <E0>1000<E1> refreshed like Heaven's dew
560	<E0>10001010<E1> if the caddies grin
561	Sustained <E0>10<E1> guide by thee
562	Where he had often <E0>0100<E1>
563	Resting there pay tribute in <E0>1010<E1> of tears
564	<E0>011<E1> to Him who evermore
565	With pipes and <E0>10010<E1> while on peg below
566	Flammarion and Kelvin <E0>100<E1> Herschel every one
567	That <E0>10<E1> prayed on that hellion ship
568	<E0>10100<E1> my Kate when life was new
569	Shook from her plumes <E0>1<E1> downy shower
570	I hear the cries of lives that rage and <E0>100<E1>
571	That <E0>100<E1> of thee awakeneth
572	<E0>100010<E1> all the tenderness that only love can know
573	And hectic autumn came <E0>100010<E1> its charm
574	<E0>10<E1> give kind Dulness memory and rhyme
575	My life into <E0>10100<E1> and follow so
576	That was <E0>100010<E1> and that was England
577	To <E0>010100<E1> laden Summer and the orb
578	There the wild <E0>0010010<E1> danced away
579	<E0>1010<E1> undreamed and forms beyond compare
580	<E0>10<E1> branch of honor flower of chevalrie
581	The sad moon poured <E0>10100<E1> into their eyes
582	He never spoke cept when <E0>1010<E1> to
583	And teem and suffer <E0>1100<E1> sound
584	Where are the songs I <E0>100010<E1> know
585	In one low <E0>01010<E1> cry
586	What of vile dust <E0>1<E1> preacher said
587	<E0>100<E1> a hearty divine
588	Now man <E0>10101000<E1> Life it is all delite
589	With mediums and prophetic chairs and crickets with a <E0>110<E1>
590	<E0>10<E1> those of learn'd philologists who chase
591	<E0>10<E1> I looked from my labour content
592	There could be <E0>1000100<E1> glad as mine
593	Found and followed like <E0>10<E1> an hour
594	It frisks and <E0>0100<E1> now here now there
595	And <E0>10<E1> the knowledge that our feet have trod
596	And the way that he gambolled <E0>100<E1>
597	And farther <E0>10<E1> and everywhere
598	A faint wind stirs and I drift <E0>110<E1>
599	Shall wear the steadfast record from the <E0>100<E1>
600	To sit and sew with eyes <E0>10010<E1>
601	And <E0>1000<E1> the cloud
602	<E0>10<E1> were very happy there
603	<E0>10<E1> the blessing of the captive
604	Survey <E0>101<E1> stranger in the painted wave
605	And all our longings lie within <E0>10100<E1>
606	Burst o'er his banks and <E0>101001010<E1> led
607	Most sweet when <E0>1010<E1> least were play'd
608	And pressed across the <E0>1011000010<E1> now their bones are laid
609	That I <E0>10100<E1> the stir and scent
610	When he <E0>10<E1> got his promis'd fee
611	And perish in such fierce <E0>100<E1>
612	Now <E0>10<E1> brothers call from the bay
613	Hers is the passion that no <E0>10<E1> shall drain
614	And their cheer will ring the <E0>10100<E1>
615	A heron's far flight to a roost <E0>1010<E1>
616	And I will make <E0>10100<E1> of roses
617	Lowly louted the boys and <E0>1011<E1> maidens all courtesied
618	It's ma leg I'm <E0>0100<E1> thinkin it's aff at the knee
619	Lives my friend because I <E0>10<E1> him still
620	And whilk of them <E0>100<E1> most quaintise
621	<E0>1000<E1> Hys Fyve Rules
622	By record of <E0>1<E1> well filled past
623	Good <E0>10110<E1> be known
624	<E0>10000100<E1> of mist so starkly bold and clear
625	A humble and a <E0>100100<E1> heart
626	<E0>10<E1> the great volcano flings
627	<E0>10<E1> grow the great fantastic flowers
628	<E0>10<E1> Virgins seem no longer vain
629	You wanted to be <E0>01011001<E1>
630	As if it could not <E0>10<E1>
631	Minnie's <E0>1110<E1> was spotless
632	You meaner beauties of the <E0>100<E1>
633	If <E0>10<E1> dream rudely shaken
634	I <E0>100<E1> prepared to make
635	A myriad rays of <E0>100<E1> may be
636	For <E0>10<E1> fancy some mistake
637	And may the countenance of <E0>1100<E1> King
638	Perhaps subject <E0>10<E1> doubts and fears
639	But <E0>101<E1> days of golden dreams had perished
640	<E0>100<E1> time is setting with me Oh
641	Our <E0>100101100<E1> till the last one died
642	Knows and yet knows <E0>10010<E1> thy healing
643	Squats <E0>100<E1> and silent for a season
644	Liv'st and will live like the <E0>0100100<E1> of Love
645	Are days <E0>10<E1> God's good giving
646	Took <E0>1010<E1> aide de camp an ass
647	Little Tim Trotter <E0>1010<E1> in his bed
648	Cassandra's gift she was <E0>1010<E1> than they
649	<E0>100<E1> ever the deathless solace left
650	CLARA winds <E0>10100<E1> and sings with Brackenburg
651	<E0>10<E1> the hound that followed slowly
652	But Snbiorn looked aloft and <E0>10<E1>
653	Some slow <E0>1010100<E1> patterns must be wrought
654	<E0>10<E1> then would gladly wish my drum
655	All tumble <E0>100<E1> and weird and broken
656	He ordered all things <E0>101<E1> busy care
657	This and the next eleven <E0>1000100<E1> not in MS
658	<E0>1010<E1> pickin your men for a fight
659	From the world s glare <E0>1010<E1> sweet vale retired
660	Paths leading from light into <E0>100010<E1>
661	<E0>10<E1> you search for new adventures
662	Then was I led to vengeful monks <E0>10100<E1>
663	The boxer in his <E0>0100<E1> may laugh
664	<E0>10<E1> my dear and tender
665	I <E0>100<E1> that you and me
666	Violets <E0>100<E1> leaves of vine
667	Its <E0>0100<E1> in the garden
668	<E0>1<E1> voiceless Form he chose to feign
669	Of Phillis too I <E0>100<E1> bereft
670	AIR Contented <E0>10<E1> am
671	In <E0>10<E1> great Butler the least crime
672	And whatever I think of them and their <E0>1000<E1>
673	The maid announced the meal <E0>10<E1> tones
674	This is <E0>1101<E1> faith and full of chear
675	Of their great <E0>110<E1> Lord and King
676	Which is played upon the <E0>10100<E1>
677	Thus <E0>1001100<E1> my love from being cruel
678	<E0>10<E1> loitering o'er her tea and cream
679	This moment <E0>10<E1> happy spell
680	In <E0>10100<E1> take a sudden blow
681	But each a soul of knowledge to <E0>0100100<E1>
682	With lifted <E0>1000<E1> and hopes divine
683	Who's that said <E0>10<E1> beats there
684	Lilting <E0>1<E1> same low lullaby again
685	es B <E0>10<E1> for the span
686	Of the Maid <E0>10<E1> fair
687	When <E0>1<E1> waggoners were down below
688	Stand <E0>11<E1> overlook'd our favourite elms
689	When <E0>10<E1> have given up struggling
690	Like <E0>10<E1> fellers does
691	He's only singing <E0>10<E1>
692	And sometimes like a gleaner thou <E0>1000100<E1>
693	Now wait <E0>100<E1> bolder deadlier crimes
694	To let the <E0>1000<E1> sleep
695	And there was <E0>10<E1> to heed
696	<E0>10<E1> are you doing farmer pray
697	The freedom of <E0>100<E1> birthright
698	Of <E0>0100<E1> and love and priceless lore
699	Might not aught outlive their trustless <E0>0100<E1>
700	<E0>100<E1> yellow balls not Typhon had withstood
701	<E0>1010<E1> appointed work from morn till even
702	To decorate the <E0>10010<E1> shaft that should
703	<E0>010101<E1> in the mind serene
704	Deeper <E0>10<E1> speech our love stronger than life our tether
705	Fretting fiercely <E0>100<E1> its narrow bounds
706	Had led our <E0>1001000<E1> sires
707	With simple air <E0>10<E1> breathed the prayer
708	But why these useless <E0>010000<E1> renew
709	And warm south winds <E0>101010<E1>
710	How papa's dear eyes did <E0>0110<E1>
711	My weary aching head <E0>100100<E1> last rest
712	May <E0>10<E1> by Chili's stern example led
713	Begins to <E0>010<E1> her gentle elbow
714	Tracks let me follow <E0>100010<E1> human kind
715	Arm within arm the couples <E0>0010<E1>
716	Who half supports him he with heavy <E0>0100<E1>
717	For truth is <E0>101010<E1> however divine
718	Scarce sixteen summers had I <E0>100<E1>
719	What is it ails <E0>1010<E1> I should sing of her
720	And if your will lies there at last forsake <E0>10<E1>
721	My but I hated to speak It certainly seemed like my <E0>100100<E1>
722	<E0>10<E1> blessed Bird the earth we pace
723	But when ye least <E0>10010010<E1> sorrow's day
724	O'er all <E0>1<E1> hedge shall ramble
725	His midnight <E0>11100<E1> he told
726	To scenes of endless joy to that fair <E0>100<E1>
727	Far <E0>010<E1> our pleasing shores to go
728	Did <E0>10<E1> a Grecian joy
729	<E0>110010<E1> Jove his empire of the skies
730	And then <E0>10<E1> heaven the monarch went
731	Once more <E0>1000<E1> glory in the strife
732	<E0>101010<E1> heard him and nobody saw
733	These do I love and <E0>100<E1> alone
734	From her loved Lord the golden <E0>10<E1>
735	Like a river's race or <E0>110010<E1> sea
736	That ye come <E0>0110010<E1> Thok's iron wood
737	Your face and <E0>10110010<E1> like a cat
738	Were singing in your heart <E0>110<E1>
739	<E0>1010<E1> she tried Alas she did not know
740	On silent <E0>100<E1> feet wending
741	Love's last victorious stand <E0>110<E1> the rout
742	Thy glory was the <E0>1000<E1> that died
743	I know what beauty is <E0>10<E1> thou
744	<E0>010100<E1> of vales is the Vale of Shanganah
745	To youthful hearts to <E0>1001<E1> anywhere
746	Knowing not the <E0>100<E1> of sorrow
747	And keep <E0>10100<E1> forlorn in bondage pent
748	Up you <E0>0010<E1> to draw him near
749	to thee and <E0>10<E1>
750	<E0>1<E1> Greek soul grown effeminate
"""

prompt_text_1000 = instructions + """
751	And from your awe they <E0>10011010<E1> take
752	Sleep <E0>110<E1> one sleep
753	With nature's <E0>1110<E1> and arts unknown before
754	Wolves shed their <E0>100<E1> and dragons scales
755	But evil things in robes of <E0>1010<E1>
756	They call'd it hectic <E0>0101<E1> fiery flush
757	The first fruits of our new <E0>1111<E1>
758	Of happier days <E0>10<E1> hand
759	Looketh down a carven face from <E0>1001<E1> gilded wall
760	New <E0>001000<E1> from their eager tone
761	With the blessings Heaven <E0>10<E1> lent
762	<E0>10<E1> fresh flowers upon a grave
763	Then while they lingered on the span wide <E0>100<E1>
764	Arose from out the <E0>10<E1> to sail with you
765	And grapes <E0>10<E1> blue
766	Clutched only <E0>100<E1> of frosted snow
767	Where the hill dreams <E0>10<E1> aid
768	<E0>10<E1> sheep for thousands gorged by men
769	Reared in <E0>1010010<E1> home and radiant yet
770	Than any crownd <E0>10<E1>
771	I turning crept on to the hedge <E0>101000<E1>
772	<E0>10<E1> deftly on a sable hem
773	O'er the decaying embers <E0>010<E1>
774	<E0>11000<E1> world legions mustering his poor clan
775	That <E0>101000<E1> lightly floated o'er his brow
776	The men that lost their <E0>10000<E1>
777	Would sate at least my passing lusts and did not <E0>101010<E1>
778	With <E0>10000<E1> and pallid features
779	But now gramercy he <E0>10<E1> dead
780	But gazed <E0>1100<E1> that hell born light
781	The young <E0>100<E1> gay declining Appia flies
782	World that knows <E0>11010<E1> all our sweet gladness
783	<E0>100<E1> still they gaze upon the ocean's hem
784	Creator and sustainer of <E0>1<E1> world
785	Beside <E0>1<E1> Burgomeister's Well
786	Alone <E0>100<E1> heard the steed's patroling tramp
787	snow white baby with big blue <E0>100<E1>
788	<E0>10<E1> rose up with the sun
789	But <E0>0101<E1> to the forest fly
790	Than dance about like you vain <E0>10<E1>
791	The votaries of the Prophet's <E0>100<E1>
792	Ah that eye has <E0>1100<E1> many
793	What custom offer'd <E0>10<E1> their dues
794	Of treasures rare <E0>10<E1> joys that thrill
795	One much above me vow'd <E0>10<E1> love and truth
796	But <E0>101<E1> curse God cleared my sight
797	And <E0>10<E1> you meet two SHUFFLING rooks
798	That rolled the waves of Nature back <E0>100<E1> cast
799	If <E0>101010<E1> well served you must serve yourself and moreover
800	And every <E0>1000<E1> expressively
801	And <E0>1010<E1> course attire which I now weare
802	<E0>1001<E1> houses you say are of jasper cut
803	Or <E0>0101<E1> spoken words of praise
804	And here thine aspen <E0>11<E1>
805	And through <E0>10<E1> leaves the robins call
806	In the hedge <E0>001110<E1> to the stream
807	That wrought to have it so <E0>10010<E1> replied
808	Who wears <E0>1<E1> robe is my Medusa still
809	And now with a wave his cloak <E0>0100010<E1>
810	Lay smooth and well polished <E0>10101100<E1> sticks
811	Should I a jot the better <E0>10<E1>
812	Made <E0>10<E1> of white thorn neatly interwove
813	Never a man of us faltered or <E0>010000101<E1> fire of the fray
814	<E0>10<E1> stern lips laugh at last
815	Was such a grief as <E0>1100<E1> be
816	Come let us <E0>100<E1> his bed
817	Where the <E0>100<E1> is all moss covered
818	The twinkling lights and <E0>110<E1> of men
819	<E0>10<E1> the summer wind goes droning o'er the sun bright seas
820	<E0>101<E1> cold creeps as the fire dies at length
821	Where faces <E0>1001010<E1> where eyelids are dewless
822	Sun illumined and white on <E0>110010<E1> verge of the ocean
823	Let this thy <E0>1010110<E1> atone for all
824	Most well <E0>11000<E1> little Prince
825	What sighs re echo'd to <E0>1010010<E1> breath
826	Of the <E0>01001<E1> lassies
827	Each minstrel of <E0>1<E1> heavenly choir
828	Here shall great Dulman <E0>10<E1> alone
829	With fleurs de lis <E0>10<E1> shone and scourged once more
830	The grandest the purest even thou hast yet <E0>100<E1>
831	Still round <E0>1100<E1> his way he takes
832	Where they <E0>10110<E1> intertwining
833	Took <E0>10<E1> in gently
834	Of <E0>100<E1> remember'd well
835	And the spider house went floating torn and tattered <E0>0101100<E1>
836	Holds <E0>10<E1> unthinking multitude enthrall'd
837	Artist in plots projector <E0>10<E1> panics he used and despised
838	So lovely <E0>100<E1> the gifts she brings
839	Rends <E0>11000100<E1> with ruin void of ruth
840	What signifies <E0>10<E1> cry
841	Whom woe and death had <E0>11100<E1>
842	Where the rocks are all bare an the turf <E0>10101010<E1>
843	They are dragging me down <E0>10<E1> hell
844	That know <E0>10100<E1> are known of me
845	I spoke of this I spoke of <E0>10<E1>
846	Of heliotrope <E0>10<E1> hangs the wall
847	Drinking the last deep light he watched it <E0>10<E1>
848	And <E0>10<E1> the fulfilling sense
849	And this my seeing is <E0>100<E1> weak
850	Dear God how great how good <E0>101000<E1>
851	What Out of danger <E0>101<E1> slighted Dame
852	They ain't no doubt about <E0>10<E1>
853	Dare ye claim sonship <E0>10100<E1> heavenly Sire
854	Clouds of affection from <E0>100101<E1> eyes passion
855	To toss the baby <E0>10<E1> the air
856	Of some land circled <E0>100<E1>
857	And <E0>100110100<E1> smokes with a score of towns
858	While in their clear <E0>100<E1> flashes
859	And where mid <E0>1010001000<E1> great sunsets burn
860	Pictures these of <E0>1<E1> Passing Show
861	<E0>10100<E1> his pipe unto his lips and blew
862	To hear <E0>10<E1> in the vernal breeze
863	England hath need of thee she <E0>101<E1> fen
864	I turn and ask <E0>10100<E1> What then
865	Just when you feel It <E0>01010100<E1> your hair
866	Watch with me men women <E0>100010010<E1> dear
867	An <E0>10<E1> his toes an croon
868	Tiny sandpipers and the huge <E0>1110<E1> seas
869	How He defers how <E0>10010<E1> is to die
870	<E0>101000<E1> through many thoughts and memories ranged
871	I ll ring <E0>10<E1> bells
872	Of little girl and little <E0>10<E1>
873	Yet should <E0>10110<E1> huge arise
874	Purge and disperse that I may see and <E0>10<E1>
875	E'en their great courage <E0>01000<E1> in vain
876	Was the spirit <E0>10<E1> passed away
877	<E0>1<E1> blithest bird of merry May
878	To morrow may bring us <E0>1<E1> halter
879	<E0>10<E1> cerebration that you think are thoughts
880	Never let <E0>100<E1> flames expire
881	How easily He <E0>1000<E1> the tides
882	Where my world <E0>11<E1> head forever lies
883	Think you <E0>100<E1> felt no charms
884	Love I marvel <E0>10<E1> you are
885	Or gather <E0>1001<E1> wall or make carouse
886	With what wild <E0>0101<E1> British set on fire
887	We're the men of Magersfontein we're the men of Spion <E0>100<E1>
888	Just as of <E0>1000<E1>
889	Then his face was somewhat browner and <E0>100100100<E1> firmer set
890	Hears ever the notes <E0>1010<E1> ever they swell subside
891	<E0>101<E1> Summer sun to his bright home run
892	It <E0>100<E1> her brother and her gentle heart
893	<E0>1010<E1> my threshold happy hour
894	You've set around in <E0>10100<E1> restawraw a year or more
895	I would lend but I would <E0>1001010<E1>
896	O the bolt and flash of doom Who trusts your <E0>100<E1>
897	One on and one a <E0>1010<E1> i my hand
898	Down swept the Lord of <E0>11100<E1>
899	Enter PRINCE <E0>0100100<E1> several of his retainers
900	They say So passes man's <E0>0100<E1> year
901	Some lightly o'er the <E0>10000<E1> skim
902	And make a throne for him within their <E0>0100<E1>
903	Where partial fortune <E0>11<E1> deign'd to smile
904	My dead my dead what words are so helpless <E0>1010<E1>
905	The days when I rode by moors and <E0>001000<E1>
906	Nature <E0>100<E1> will stupefy
907	And cruise at <E0>10<E1> ease in the climates above
908	Deep <E0>01010<E1> he hears not a word that they say
909	He and his racers were <E0>1100<E1>
910	There <E0>10001<E1> face I have not seen
911	North <E0>100<E1> south and east and west
912	Thee Baby laughing in my <E0>10000<E1>
913	To believe <E0>1100<E1> she says she needs broad'ning we conjecture
914	They had snarled at you barked at you <E0>10001010<E1> day after day
915	With visits or to beg <E0>1<E1> place
916	That if the magistrates of all the <E0>1100<E1>
917	Leaving less <E0>1100<E1> the snow
918	Through distant age saints travell'd martyrs <E0>010<E1>
919	A hymn to <E0>1010<E1> the budding year
920	Am I so blind Or thou <E0>1000101<E1> see
921	For who can write and speak <E0>1010<E1> and I
922	Duly too at <E0>0100<E1> of day
923	Telling a tale <E0>10<E1> bliss
924	But by reverse of <E0>10010<E1> chased away
925	Where <E0>100<E1> no more molest
926	Elegie upon <E0>1<E1> untimely death of the
927	Around thy walls their hell <E0>100<E1> demons led
928	Cold northern natures <E0>100<E1> perhaps
929	When passion tempted me to <E0>0100<E1>
930	<E0>1010<E1> draw near to their eternal home
931	<E0>10010<E1> goes Whoosh at the station
932	I saw the proper twinkle in your <E0>10<E1>
933	From Hackney then he <E0>10<E1> provide
934	A box of counters and a <E0>10<E1> veined stone
935	Thy shapeless <E0>100<E1> nor step nor grace
936	<E0>100<E1> left Niagara almost dry
937	His silks and his <E0>01010<E1> are dreams
938	<E0>100<E1> thou return to earth's sad scenes
939	And <E0>10010<E1> as the dews of dawn
940	Like the <E0>100<E1> ocean's fluctuating toil
941	Upon the mis'ries <E0>10<E1> befall the great
942	That word illstarred no <E0>1000<E1> has marred
943	The trumpets and the battle <E0>010100<E1> cease
944	It is a sad forgotten <E0>100<E1>
945	That matter of the <E0>100010<E1> was at Valencia too
946	For sedentary service <E0>10<E1> unfit
947	The moonbeams over Arno's vale in <E0>101010<E1> were pouring
948	When hard <E0>101111000<E1> times
949	<E0>11000<E1> them that come to woo
950	And on Patrick s Night if ye hear <E0>1<E1> pig play
951	And slily he traileth along <E0>1<E1> ground
952	That <E0>100<E1> between October sun and rain
953	Owing a royal debt and <E0>1000<E1> it
954	Was there a tear <E0>10<E1> her eye
955	Where seraphs <E0>0100<E1> them glows
956	These fields my dear Ellen I <E0>10<E1> them of yore
957	Jewels gleamed not in the tresses <E0>10<E1>
958	<E0>10<E1> them my friend my Hoel died
959	Wavy <E0>10<E1> gold and agate banded globes
960	<E0>10<E1> the wife's and half the mother's
961	And <E0>10<E1> I sought the dentist's aid
962	<E0>1100<E1> whose beauty rumors hum
963	And southern climes shall <E0>100<E1> my rhymes
964	Where are the japeries fresh or <E0>0100<E1>
965	Of <E0>100<E1> green beauty it was shorn
966	Where mighty lakes <E0>1100<E1> the fullest scope
967	But <E0>10<E1> the uncomplaining ones that wear a sorrow long
968	And I roomed <E0>10<E1> the cool of a cave
969	Aw've crept to hear her <E0>0100<E1>
970	I read his pledge of <E0>10100<E1> soon or late
971	<E0>1<E1> fruitful top is spread on high
972	Who <E0>100<E1> in sin is surely dead
973	In Tennessee and Kentucky slaves <E0>11101<E1> coalings at the forge
974	That was a word she never <E0>100<E1> to say
975	Come tell <E0>10<E1> then why
976	Doubt flies when Teucer leads <E0>1001000<E1> despair
977	With my hand I crush <E0>10<E1> up
978	<E0>10<E1> when against the tide of years
979	While none can <E0>100<E1> with kinglier pride
980	<E0>10100<E1> of the church in an angry rout
981	<E0>110<E1> my bosom swelling
982	<E0>10100<E1> it by turns and milk her dry
983	<E0>10<E1> spotless as she's bonnie O
984	Here Beauty's day doth never <E0>100<E1>
985	Him soul and self more hated <E0>1010<E1> God
986	Our <E0>110<E1> fond and fair
987	<E0>100100<E1> with joy and she comes forth to sing
988	There is no <E0>0100<E1> within the flag
989	Could she survive the <E0>1010<E1> blow
990	Yon ambient azure shell <E0>100<E1> spring to life
991	<E0>1<E1> mill pond's gloom
992	Are not <E0>10<E1> anywhere
993	<E0>10<E1> watch there with clear eyes
994	<E0>10<E1> held a cabin for ten groats a year
995	Whose <E0>0100<E1> are dripping with the grape blood sweet
996	Her own soul and another's <E0>10<E1> endure
997	Ah in this world <E0>10110<E1> guiding thread
998	But faster and <E0>101<E1> year on year
999	All words are <E0>10<E1>
1000	<E0>10<E1> beauty comes to passion's trysting ground
"""