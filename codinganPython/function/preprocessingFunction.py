import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Unduh stopwords dan wordnet jika belum diunduh
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

norm= {" dgn " : " dengan ", ' seller ': ' penjual ',' service ':' pelayanan ', ' tp ':' tapi ', ' recommended ':' rekomendasi ', ' kren ':' keren ', ' kereen ':' keren ', ' mantab ': ' keren ',' matching ':' sesuai ','happy':' senang ','original': 'asli ','ori':'asli ', "trusted" : "terpercaya", "angjaaaassss":"keren", " gue ": " saya ", "bgmn ":" bagaimana ", ' tdk':' tidak ', ' blum ':' belum ', 'mantaaaaaaaappp':' bagus ', 'mantaaap':'bagus ', ' josss ':' bagus ', ' thanks ': ' terima kasih ', 'fast':' cepat ', ' dg ':' dengan ', 'trims':' terima kasih ', 'brg':' barang ', 'gx':' tidak ', ' dgn ':' dengan ', ' recommended':' rekomen ', 'recomend':' rekomen ', 'good':' bagus ', " dgn " : " dengan ", " gue ": " saya ", " dgn ":" dengan ", "bgmn ":" bagaimana ", ' tdk':' tidak ', 
' blum ':' belum ', "quality":"kualitas", 'baguss':'bagus', 'overall' : 'akhirnya', 'mantaaaaaaaappp':' bagus ', ' josss ':' bagus ', ' thanks ': ' terima kasih ', 'fast':' cepat ', 
 'trims':' terima kasih ', 'brg':' barang ', 'gx':' tidak ', ' dgn ':' dengan ', ' real ': ' asli ', ' bnb ': ' baru ' ,
' recommended':' rekomen ', 'recomend':' rekomen ', 'good':'bagus',
'eksis ':'ada ', 'beenilai ':'bernilai ', ' dg ':' dengan ', ' ori ':' asli ', ' setting ':' atur ', " free ":" gratis ",
' yg ':' yang ', 't4 ':'tempat', ' awat ':' awet', ' mantep ':' bagus ', 'mantapp':'bagus', 
'kl ':'kalo', ' k ':' ke ', 'plg ':'pulang ', 'ajah ':'aja ', 'bgt':'banget', 'lbh ':'lebih', 'ayem':'tenang','dsana ':'disana ', 'lg':' lagi',
'pas ':'saat ', ' bnib ': ' baru ', 
' nggak ':' tidak ', 'karna ':'karena ', 'utk ':'untuk ',
' dn ':' dan ', ' mlht ':' melihat ', ' pd ':' pada ', 'mndngr ':'mendengar ', 'crita':'cerita', ' dpt ':' dapat ', ' mksh ':' terima kasih ', ' sellerrrr':' penjual', 'ori ':'asli ', ' new ':' baru ',
'sejrh':'sejarah', 'mnmbh ':'menambah ', 'sayapun':'saya', 'thn ':'tahun ', 'good':'bagus', ' awettt':' awet',
'halu ':'halusinasi ', ' nyantai ':' santai ', 'plus ':'dan ',
' ayang ':' sayang ', ' Rekomendded ':' direkomendasikan ', ' now ': ' sekarang ', 'slalu ':'selalu ', 'photo ': 'foto ', 'slah ':'salah ', 'krn':'karena', ' ga ':' tidak ', 'ok ':'oke ', ' meski':' mesti', ' para ':'parah', ' nawarin':' menawari', 'socmed':'sosial media',
' sya ':' saya ', 'siip':'bagus', ' bny ':' banyak ', ' tdk ':' tidak ', ' byk ':' banyak ', 
' pool ':' sekali ', " pgn ":" ingin ", " gue ":" saya ", " bgmn ":" bagaimana ", " ga ":" tidak ", 
" gak ":" tidak ", " dr ":" dari ", " yg ":" yang ", " lu ":" kamu ", " sya ":" saya ", 
" lancarrr ":" lancar ", " kayak ":" seperti ", " ngawur ":" sembarangan ", " k ":" ke ", 
" luasss ":" luas ", " sy ":" saya ", " thn ":" tahun ", " males ":" malas ",
" tgl ":" tanggal ", " lg ":" lagi ", " bgt ":" banget ",' gua ':' saya ', '\n':' ', ' tpi ':' tapi ', ' standar ':' biasa ', ' standart ': ' biasa ', ' sdh ':' sudah ', ' n ':' dan ', ' gk ': ' tidak ', ' mengecwakan ':' mengecewakan ', ' d ':' di ', ' approved':' setuju', 'ademmmm ':'adem ', ' g ':' tidak ', ' gak ':' tidak ', 'gak ':'tidak ', ' cpt ':' cepat ', ' ku ':' aku ', ' design ':' desain ', ' purple ':' ungu ', 'bgus ':'bagus ', ' bgus ':' bagus ', ' stock ':' stok ', ' cumaa ':' hanya ', ' lmyan ':' lumayan ', ' gtu ':' gitu ', ' jatoh ':' jatuh ', ' koq ':' kok ', 'bnyk ':'banyak ', ' bnyk ':' banyak ', 'lucuuu ':'lucu ', ' lucuuu ':' lucu ', ' udh ':' udah ', ' mantaaaaaap ':' mantap ', ' check ':' cek ', ' mintiib ':' mantap ',
' bbrp ':' beberapa ', 'bbrp ':'beberapa ', 'sy ':'aku ', ' sy ':' saya ', ' pengiirman ':' pengiriman ', 'mantull ':'mantap betul ', 'bbrp ':'beberapa ', ' bbrp ':' beberapa ', ' brp ':' berapa ', 'brp ':'berapa ', ' makasiih ':' makasih ', 'makasiih ':'makasih ', 'napa ':'kenapa ', ' napa ':' kenapa ', ' jdnya ':' jadi ', 'jdnya ':'jadi ', ' sm ':' sama ', 'sm ':'sama ',
'nyobain ':'coba ',' nyobain ':' coba ', ' nyobain':' coba', 'kecewaaaaa ':'kecewa ',' kecewaaaaa ':' kecewa ', ' kecewaaaaa':' kecewa', 'sukak ':'suka ',' sukak ':' suka ', ' sukak':' suka', 'resp ':'respon ',' resp ':' respon ', ' resp':' respon', 'bangetttttttt ':'banget ',' bangetttttttt ':' banget ', ' bangetttttttt':' banget', 'tsb ':'tersebut ',' tsb ':' tersebut ', ' tsb':' tersebut', 'mantaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaap ':'mantap ',' mantaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaap ':' mantap ', ' mantaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaap':' mantap', 'cakepppp ':'bagus ',' cakepppp ':' bagus ', ' cakepppp':' bagus', 'keceee ':'keren ',' keceee ':' keren ', ' keceee':' keren', 'kece ':'keren ',' kece ':' keren ', ' kece':' keren', 'yng ':'yang ',' yng ':' yang ', ' yng':' yang', 'usa ':' ',' usa ':'  ', ' usa':' ', 'baguusss ':'bagus ',' baguusss ':' bagus ', ' baguusss':' bagus', 'disc ':'diskon ',' disc ':' diskon ', ' disc':' diskon', 'hehe ':' ',' hehe ':'  ', ' hehe':' ',
'bb ':'berat badan ',' bb ':' berat badan ', ' bb':' berat badan', 'tb ':'tinggi badan ',' tb ':' tinggi badan ', ' tb':' tinggi badan', 'kg ':'kilogram ',' kg ':' kilogram ', ' kg':' kilogram', 'bangettt ':'banget ',' bangettt ':' banget ', ' bangettt':' banget', 'jd ':'jadi ',' jd ':' jadi ', ' jd':' jadi', 'me ':'aku ',' me ':' aku ', ' me':' aku', 'gpp ':'gapapa ',' gpp ':' gapapa ', ' gpp':' gapapa', 'naikin ':'naik ',' naikin ':' naik ', ' naikin':' naik', 'lu ':'kamu ',' lu ':' kamu ', ' lu':' kamu', 'pny ':'punya ',' pny ':' punya ', ' pny':' punya', 'cepatt ':'cepat ',' cepatt ':' cepat ', ' cepatt':' cepat', 'banyakin ':'banyak ',' banyakin ':' banyak ', ' banyakin':' banyak', 'thx ':'makasih ',' thx ':' aku ', ' thx':' aku', 'dibeliin ':'beli ',' dibeliin ':' beli ', ' dibeliin':' beli', 'smpe ':'sampai ',' smpe ':' sampai ', ' smpe':' sampai', 'udh ':'udah ',' udh ':' udah ', ' udh':' udah', 'gmbr ':'gambar ',' gmbr ':' gambar ', ' gmbr':' gambar', 'bnykkk ':'banyak ',' bnykkk ':' banyak ', ' bnykkk':' banyak', 'dtg ':'datang ',' dtg ':' datang ', ' dtg':' datang', 'pcs ':'pieces ',' pcs ':' pieces ', ' pcs':' pieces', 'kermh ':'rumah ',' kermh ':' rumah ', ' kermh':' rumah', 'respononsif ':'responsif ',' respononsif ':' responsif ', ' respononsif':' responsif', 'seller ':'penjual ',' seller ':' penjual ', ' seller':' penjual', 'bhn ':'bahan ',' bhn ':' bahan ', ' bhn':' bahan',
'spt ':'seperti ',' spt ':' seperti ', ' spt':' seperti', 'lamaaa ':'lama ',' lamaaa ':' lama ', ' lamaaa':' lama', 'jgn ':'jangan ',' jgn ':' jangan ', ' jgn':' jangan', 'dimodif ':'modifikasi ',' dimodif ':' modifikasi ', ' dimodif':' modifikasi', ' pic ':' gambar ', ' tdi ':' tadi ', ' kyk ':' mirip ', ' seller ':' penjual ', ' skrg ':' sekarang ', ' nyesal ':' menyesal ', ' bagusss ':' bagus ', ' buy ':' beli ', ' kringet ':' keringat ', 'wkwk ':' ', ' wkwk ':' ', ' wkwk':' ', ' teball ':' tebal ', ' maksa ':' paksa ',
'plis ':'tolong ',' plis ':' tolong ', ' plis':' tolong', 'karenaa ':'karena ',' karenaa ':' karena ', ' karenaa':' karena', 'dsni ':'disini ',' dsni ':' disini ', ' dsni':' disini', 'beranrakan ':'berantakan ',' beranrakan ':' berantakan ', ' beranrakan':' berantakan', 'pakek ':'pakai ',' pakek ':' pakai ', ' pakek':' pakai', 'pdhl ':'padahal ',' pdhl ':' padahal ', ' pdhl':' padahal', ' kereeen ':' keren ', ' ttp ':' tetap ', ' bngt ':' banget ',
' lmyn ':' lumayan ', 'ujurannya ':'ukuran ', ' ujurannya ':' ukuran ', ' ujurannya':' ukuran', 'sblmnya ':'sebelum ', ' sblmnya ':' sebelum ', ' sblmnya':' sebelum', 'trnyta ':'ternyata ', ' trnyta ':' ternyata ', ' trnyta':' ternyata', ' hpus ':' hapus '}

def normalisasi(text):
  for i in norm:
    text = text.replace(i, norm[i])
  return text

def clean(text):
  text = text.strip()
  text = text.lower()
  text = re.sub(r'[^a-zA-Z]+', ' ', text)
  return text

def labeling(rating):
    rating = str(rating)
    if rating == '4' or rating == '5':
       return 'Positif'
    else :
       return 'Negatif'

def tokenisasi(text):
    return text.split() 
    
def stopword(text):
    stop_words = set(stopwords.words('indonesian'))
    words = text.split()
    filtered_words = [word for word in words if word.casefold() not in stop_words]
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

def stemming(text):
    stemmer = StemmerFactory().create_stemmer()
    text = ' '.join(text)
    stemmed_text = stemmer.stem(text)
    return stemmed_text

# Definisi fungsi filter_tokens_by_length
def filter_tokens_by_length(dataframe, column, min_words, max_words):
    # Tokenisasi kata
    words_count = dataframe[column].astype(str).apply(lambda x: len(x.split()))
    # Membuat filter untuk jumlah kata
    mask = (words_count >= min_words) & (words_count <= max_words)
    # Mengaplikasikan filter ke DataFrame
    df = dataframe[mask]
    return df

