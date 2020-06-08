/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package identificadorderosto;

import Catalano.Imaging.FastBitmap;
import Catalano.Imaging.Filters.Photometric.SelfQuocientImage;
import Catalano.Imaging.Texture.BinaryPattern.ImprovedLocalBinaryPattern;
import Catalano.Imaging.Tools.SpatialHistogram;
import Catalano.MachineLearning.Classification.IClassifier;
import Catalano.MachineLearning.Classification.MulticlassSupportVectorMachine;
import Catalano.MachineLearning.Dataset.DatasetClassification;
import Catalano.MachineLearning.Performance.HoldoutValidation;
import Catalano.Statistics.Kernels.ChiSquare;
import java.io.File;
import java.io.IOException;

/**
 *
 * @author David
 */
public class IdentificadorDeRosto {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        
        File folder = new File("D:\\PDI\\IdentificadorDeRosto\\att_faces");
        File[] lisOfFiles = folder.listFiles();
        double m[][] = new double [400][6*6*511] ;
        int l[] = new int[400];
        int id=0;
        int leg=0;
        
        
        for (File file : lisOfFiles) {
            File[] files = file.listFiles();
            
            for (int i = 0; i < files.length; i++) {
                
                //colocar a legenda no vetor
                l[id]=leg;
                
                //ler a imagem imagem em pgm
                int[][] image = PGMIO.read(new File(files[i].getPath()));
                //converte a imagem para fastbitmap
                FastBitmap fb = new FastBitmap(image);
                
                SelfQuocientImage sqi = new SelfQuocientImage();
                sqi.applyInPlace(fb);
                
                //usa as ferramentas para tirar as caracteristicas
                SpatialHistogram sh = new SpatialHistogram(6, 6);
                int[] f = sh.Compute(fb, new ImprovedLocalBinaryPattern());
                
                //conversão de int pra double
                double[] feat = new double[f.length];
                for(int j=0; j<f.length; j++) {
                    feat[j] = f[j];
                }
                
                m[id++]=feat; 
            }
            
            leg++;
            
        }
        
        
        DatasetClassification dc = new DatasetClassification("dataset", m, l);
        //dc.Normalize();
        
        IClassifier clas = new MulticlassSupportVectorMachine(new ChiSquare(), 1, dc.getNumberOfClasses());
                
        HoldoutValidation val = new HoldoutValidation();
                
        double p = val.Run(clas, dc);
                
        System.out.println(p);
        

        
    }
    
}
