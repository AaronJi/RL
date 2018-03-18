use strict;

#
#   All right reserved, Microsoft Research Asia
#   Written by Wenying Xiong and Tie-Yan Liu
#   Contact: tyliu@microsoft.com
#

my $argc = $#ARGV+1;
if($argc != 4)
{
		print "Invalid command line.\n";
		print "Usage: perl eval.pl argv[1] argv[2] argv[3] argv[4]\n";
		print "argv[1]: testset file \n";
		print "argv[2]: prediction file\n";
		print "argv[3]: output file\n";
		print "argv[4]: flag. If flag equals 1, output the evaluation results per query; if flag equals 0, simply output the average results.\n";
		exit -1;
}
my $test = $ARGV[0];
my $testprediction = $ARGV[1];
my $output = $ARGV[2];
my $flag = $ARGV[3];
if($flag != 1 and $flag != 0)
{
	print "Invalid command line.\n";
	print "Usage: perl eval.pl argv[1] argv[2] argv[3] argv[4]\n";
	print "Flag should be 0 or 1\n";
	exit -4;
}
my @start;
my @value;
my @label;
my @precision;
my @MeanPrecision;
my @two_count;
my @one_count;
open(OUT, ">$output");
open_file();
PrecisionAtN();
for(my $i = 0; $i < 16; $i ++)
{
	$MeanPrecision[$i] /= $#start;
}
print OUT "precision:";
for(my $i = 0; $i < 16; $i ++)
{
	print OUT "$MeanPrecision[$i]\t";                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
}
print OUT "\n\n";
MeanAP();
compute();
close(OUT);

#open feature file and score file, initialization
sub open_file
{
	if(!open(IN1, "$test"))
	{
		print "Invalid command line.\n";
		print "Usage: perl eval.pl argv[1] argv[2] argv[3] argv[4]\n";
		print "Open \"$ARGV[0]\" failed.\n";
		exit -2;
	}
	if(!open(IN2, "$testprediction"))
	{
		print "Invalid command line.\n";
		print "Usage: perl eval.pl argv[1] argv[2] argv[3] argv[4]\n";
		print "Open \"$ARGV[1]\" failed.\n";
		exit -3;
	}
	my @lns = <IN1>;
	close(IN1);
	my $str = "0";
	my $k = 0;
	my $i;
	for($i = 0; $i < @lns; $i++)
	{
		chomp($lns[$i]);
		#print OUT "$lns[$i]\n";
		my $where1 = index($lns[$i], "qid:");
		my $where2 = index($lns[$i], " ", 3);
		my $s = substr($lns[$i], $where1, $where2-$where1);
		if($s ne $str)
		{
			$start[$k] = $i;					#collect differnt query ids
			$str = $s;
			$k ++;
		}
	}
	my @lns = <IN2>;
  close(IN2);
	open(IN1, "$test");
	my @lns1 = <IN1>;
	close(IN1);
	my $j = @start;
	$start[$j] = $i;
	for(my $i = 0; $i < @start-1; $i ++) 
	{
		my @tt;
		my @mark;
		my $k = 0;
		for(my $j = $start[$i]; $j < $start[$i+1] ; $j ++)
		{
			
				chomp($lns[$j]);
				#print OUT "$lns[$j]\n";
				$tt[$k] = $lns[$j];
				
				$mark[$k] = $k;
				$k ++;			
				
		}	
		#print OUT "\n";	
		my @double = doubleSort(\@tt, \@mark);
		for(my $j = 0; $j < $start[$i+1]-$start[$i]; $j++)
		{
			$tt[$j] = $double[0][$j];
			$mark[$j] = $double[1][$j];
		}
		#for(my $j = 0; $j < $start[$i+1]-$start[$i]; $j++)
		#{
		#	print OUT "$mark[$j]\n";
		#}
		my $key;
		for(my $j = 0; $j < $start[$i+1]-$start[$i]; $j ++)
		{
			$value[$i][$j] = $tt[$j];
		}
		for(my $j = 0; $j < $start[$i+1]-$start[$i] ; $j ++)	#store the label
		{
			my $k;
			$k = $start[$i] + $mark[$j];
			chomp($lns1[$k]);
			my $where = index($lns1[$k], " ");
			my $s = substr($lns1[$k], 0, $where);
			#print OUT "$lns1[$k]\n";
			$label[$i][$j] = $s;                           
		}	
	}	
	return (@value, @start, @label);
}

sub doubleSort
{
	my ($ref_tt, $ref_mark) = @_;
	my @tt = @{$ref_tt};
	my @mark = @{$ref_mark};
	my $i;
	my $j;
	for($i = 0; $i < @tt-1; $i ++ )
	{
		for($j = 1; $j < @tt - $i; $j ++)
		{
			if($tt[$j-1]<$tt[$j])
			{
				my $temp1 = $tt[$j-1];
				$tt[$j-1] = $tt[$j];
				$tt[$j] = $temp1;
				my $temp2 = $mark[$j-1];
				$mark[$j-1] = $mark[$j];
				$mark[$j] = $temp2;
			}
		}
	}	
	my @double;
	for($j = 0; $j < @tt; $j ++)
	{
		$double[0][$j]=$tt[$j];
		$double[1][$j]=$mark[$j];
	}
	return @double;
}

sub PrecisionAtN
{
	for(my $i = 0; $i < 16; $i ++)
	{
		$MeanPrecision[$i] = 0;
	}
	for(my $i = 0; $i < @start-1; $i ++)
	{
		my $j;
		
		if($label[$i][0] == 0)
		{
			$precision[$i][0] = 0;
		}
		else
		{
			$precision[$i][0] = 1;
		}
		if($flag == 1)
		{
			print OUT "precision of query$i:\t";
		}
		for($j = 1; $j < $start[$i+1]-$start[$i]; $j ++)
		{
			
			if($label[$i][$j] == 0)
			{
				$precision[$i][$j] = $precision[$i][$j-1];
			}
			else
			{
				$precision[$i][$j] = $precision[$i][$j-1] + 1;
			}
			#print OUT "$label[$i][$j]\t";
			$precision[$i][$j-1] /= $j;
			if($flag == 1 && $j <= 16)
			{
				print OUT "$precision[$i][$j-1]\t";
			}
			if($flag == 1 && $j == 16)
			{
				print OUT "\n";
			}
			$MeanPrecision[$j-1] += $precision[$i][$j-1];
		}
		$precision[$i][$j-1] /= $j;
		$MeanPrecision[$j-1] += $precision[$i][$j-1];
	}
	return (@MeanPrecision, @precision);
}

sub MeanAP
{
	my @AveragePrecision;
	for(my $i = 0; $i < @start-1; $i ++)
	{
		$two_count[$i] = 0;
		$one_count[$i] = 0;
		for(my $j = 0; $j < $start[$i+1]-$start[$i]; $j ++)
		{
			if($label[$i][$j] eq "2")
			{
				$two_count[$i] ++;
				$AveragePrecision[$i] += $precision[$i][$j];
			}
			elsif($label[$i][$j] eq "1")
			{
				$one_count[$i] ++;
				$AveragePrecision[$i] += $precision[$i][$j];
			}
			
		}
		my $d = $two_count[$i] + $one_count[$i];
		if($d == 0)
		{
			$AveragePrecision[$i] = 0;
		}
		else
		{
			$AveragePrecision[$i] /= ($two_count[$i]+$one_count[$i]);
		}
		if($flag == 1)
		{
			print OUT "Map of query$i:\t$AveragePrecision[$i]\n";
		}
	}
	my $MeanAP = 0;
	for(my $i = 0; $i < @start; $i ++)
	{
		$MeanAP += $AveragePrecision[$i];
	}
	$MeanAP /= $#start;
	print OUT "MAP:\t$MeanAP\n\n";
	return (@two_count, @one_count);	
}


sub computeG
{
	my @G;
	for(my $i = 0; $i < 16; $i ++)
	{
		$G[$i] = 0;
		my $k;
		$k = $label[$_[0]][$i];
		my $l = exp($k*log(2))-1;
		$G[$i] = $l;
	}
	return @G;
}


sub computeBestG
{
	my @BestG;
	my $j;
	my $i = $_[0];
	#print "$two_count[$i]\t$one_count[$i]\n";
	for($j = 0; $j < 16; $j ++)
	{
		$BestG[$j] = 0;
	}
	if($two_count[$i] >= 16)
	{
		for($j = 0; $j < 16; $j ++)
		{
			$BestG[$j] = 3; 
		}
	}
	elsif($two_count[$i] + $one_count[$i] >= 16)
	{
		for($j = 0; $j < $two_count[$i]; $j ++)
		{
			$BestG[$j] = 3;
		}
		for($j = $two_count[$i]; $j < 16; $j ++)
		{
			$BestG[$j] = 1;
		}
	}
	else
	{
		for($j = 0; $j < $two_count[$i]; $j ++)
		{
			$BestG[$j] = 3;
		}
		for($j = $two_count[$i]; $j < $two_count[$i]+$one_count[$i]; $j ++)
		{
			$BestG[$j] = 1;
		}
		#print "$BestG[$j]\t";
	}
	#print "\n";
	for($i = 0; $i < 5; $i ++)
	{
		#print OUT "$BestG[$i]\t";
	}
	#print OUT "\n";
	return @BestG;
}

sub computeCG
{
	my @G = @_;
	my @CG;
	for(my $i = 0; $i < 16; $i ++)
	{
		if($i == 0)
		{
			$CG[0] = $G[0];
		}
		else
		{
			$CG[$i] = $CG[$i-1] + $G[$i];
		}
		#print OUT "$CG[$i]\t";
	}
	#print OUT "\n";
	return @CG;
}
sub computeDCG
{
	my @DCG;
	my($ref_CG, $ref_G) = @_;
	my @CG = @{$ref_CG};
	my @G = @{$ref_G};
	for(my $j = 0; $j < 16; $j ++)
	{
		if($j < 2)
		{
			$DCG[$j] = $CG[$j];
		}
		else
		{
			my $d;
			$d = log($j+1) / log(2);
			$DCG[$j] = $DCG[$j-1] + $G[$j] / $d; 
			
		}
		#print OUT "$DCG[$j]\t";
		
	}
	#print OUT "\n";
	return @DCG;
}

sub computeNDCG
{
	my @NDCG;
	my($ref_DCG, $ref_BestDCG) = @_;
	my @DCG = @{$ref_DCG};
	my @BestDCG = @{$ref_BestDCG};
	for(my $i = 0; $i < 16; $i ++)
	{
		if($BestDCG[$i] == 0)
		{
			$NDCG[$i] = 0;
		}
		else
		{
			$NDCG[$i] = $DCG[$i] / $BestDCG[$i];
		}
		if($flag == 1)
		{
			print OUT "$NDCG[$i]\t";
		}
	}
	if($flag == 1)
	{
		print OUT "\n";
	}
	return @NDCG;
}
sub compute
{
	my @result;
	my $i;
	my $j;
	for($i = 0; $i < 16; $i ++)
	{
		$result[$i] = 0;
	}
	for($i = 0; $i < @start-1; $i ++)
	{
		my @G = computeG($i);
		my @BestG = computeBestG($i);
		my @CG = computeCG(@G);
		if($flag == 1)
		{
			print OUT "NDCG of query$i:\t";
		}
		my @BestCG = computeCG(@BestG);
		my @DCG = computeDCG(\@CG,\@G);
		my @BestDCG = computeDCG(\@BestCG,\@BestG);
		my @NDCG = computeNDCG(\@DCG,\@BestDCG);
		for($j = 0; $j < 16; $j ++)
		{
			$result[$j] += $NDCG[$j];
		}
	}
	print OUT "NDCG:\t";
	#print "$#start";
	for($i = 0; $i < 16; $i ++)
	{
		$result[$i] /= $#start;
		print OUT "$result[$i]\t";
	}
}

