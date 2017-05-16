package util;

import org.wltea.analyzer.core.IKSegmenter;
import org.wltea.analyzer.core.Lexeme;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by cluster on 2017/4/13.
 */
public class JavaUtil {

    public List<String> getSplitWords(String line){
        int wordLength = 0; //0 为所有单词长度
        List<String> words = new ArrayList<String>();
        if (line == null || line.trim().length() == 0){
            return words;
        }

        try {
            //把stopword.dic和IKAnalyzer.cfg直接拖到src的下级目录中，就可以使用去掉停用词了
            InputStream is = new ByteArrayInputStream(line.getBytes("UTF-8"));
            //false代表最细粒度切分
            IKSegmenter seg = new IKSegmenter(new InputStreamReader(is),false);

            Lexeme lex = seg.next();

            while (lex != null){
                String word = lex.getLexemeText();
                if (wordLength == 0 || word.length() == wordLength){
                    words.add(word);
                }

                lex = seg.next();
            }

        } catch (UnsupportedEncodingException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return words;
    }
}
