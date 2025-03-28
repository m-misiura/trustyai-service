import math
from typing import List, Any, Optional
from collections import namedtuple

# Similar to Apache Commons Triple
Triple = namedtuple('Triple', ['left', 'middle', 'right'])

class GKException(Exception):
    """Exception from unexpected execution of GKSketch."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class GKSketch:
    """
    Greenwald-Khanna epsilon sketch for approximate quantiles.
    
    Algorithm is based on M. Greenwald and S. Khanna. Space-efficient online computation 
    of quantile summaries. In SIGMOD, pages 58–66, 2001
    """
    
    def __init__(self, epsilon: float, xmin: float = None, xmax: float = None, d: int = None):
        """Initialize GKSketch with given parameters."""
        self.epsilon = epsilon
        
        if xmin is not None and xmax is not None and d is not None:
            self.xmin = xmin
            self.xmax = xmax
            self.numx = d
            self.summary = []
        else:
            self.summary = []
            self.numx = 0
            self.xmin = float('-inf')
            self.xmax = float('inf')
    
    def insert(self, x: float) -> None:
        """Main method to process stream."""
        if not math.isfinite(x):
            return
        compress_steps = int(math.floor(1.0 / (2.0 * self.epsilon)))
        if self.numx % compress_steps == 0:
            self.compress()
        
        self.numx += 1
        
        try:
            self.update(x)
        except GKException as e:
            raise RuntimeError(f"Unexpected execution of GKSketch: {e.message}")
    
    def compress(self) -> None:
        """Compress the summary using band-based approach."""
        if len(self.summary) < 3:
            return
        
        i = len(self.summary) - 2
        while i > 1:
            band_curr = self.find_band(i)
            band_next = self.find_band(i + 1)
            cond1 = band_curr <= band_next
            
            children = self.get_children(i)
            sum_g = self.get_subtree_sum_g(i, children)
            
            t = self.summary[i + 1]
            lhs = sum_g + t.middle + t.right
            rhs = 2 * self.epsilon * self.numx
            cond2 = lhs < rhs
            
            if cond1 and cond2:
                new_t = Triple(t.left, t.middle + sum_g, t.right)
                
                # Remove in reverse order to maintain correct indices
                self.summary.pop(i + 1)
                self.summary.pop(i)
                
                insert_index = i
                # Sort children in descending order to remove safely
                for j in sorted(children, reverse=True):
                    self.summary.pop(j)
                    insert_index = j
                
                self.summary.insert(insert_index, new_t)
                i = insert_index
            
            i -= 1
    
    def get_subtree_sum_g(self, i: int, children: List[int]) -> int:
        """Sum of g values for subtree rooted at node i."""
        g_star = self.summary[i].middle
        
        for child in children:
            g_star += self.summary[child].middle
        
        return g_star
    
    def get_children(self, i: int) -> List[int]:
        """Entries in the summary that are children of Ith element."""
        subtree = []
        delta_i = self.summary[i].right
        
        for j in range(i - 1, -1, -1):
            delta_j = self.summary[j].right
            if delta_j > delta_i:
                subtree.append(j)
            else:
                break
                
        return subtree
    
    def find_band(self, i: int) -> int:
        """Find the band for the ith element."""
        var_p = int(2 * self.epsilon * self.numx)
        delta = self.summary[i].right
        diff = var_p - delta + 1
        
        band = 0 if diff == 1 else int(math.log(diff) / math.log(2))
        
        return band
    
    def update(self, x: float) -> None:
        """Update the sketch summary with a new value."""
        if self.numx == 1:
            self.xmin = x
            self.xmax = x
            self.summary.append(Triple(x, 1, 0))
        elif x <= self.xmin:  # add as new min
            self.xmin = x
            self.summary.insert(0, Triple(x, 1, 0))
        elif x >= self.xmax:  # add as new max
            self.xmax = x
            self.summary.append(Triple(x, 1, 0))
        else:
            i = self.find_index(x)
            g_i = self.summary[i].middle
            delta_i = self.summary[i].right
            delta = g_i + delta_i - 1
            self.summary.insert(i, Triple(x, 1, delta))
    
    def find_index(self, x: float) -> int:
        """Find smallest i such that v[i-1] <= v < v[i]."""
        new_i = -1
        
        for i in range(1, len(self.summary)):
            j = i - 1
            tj = self.summary[j]
            vj = tj.left
            
            if x >= vj:
                ti = self.summary[i]
                vi = ti.left
                
                if x < vi:
                    new_i = i
                    break
        
        if new_i < 0:
            raise GKException(f"Could not find insertion location for {x} in GKsketch values in range [{self.xmin},{self.xmax}]\".")
        
        return new_i
    
    def rank(self, x: float) -> int:
        """Find approx rank of x."""
        if x <= self.xmin:
            return 1
        
        if x >= self.xmax:
            return self.numx
        
        index = self.find_index(x)
        rank = self.get_min_rank(index) - 1
        
        return rank
    
    def get_min_rank(self, index: int) -> int:
        """Get minimum rank at index."""
        if index == 0:
            return 1
        
        rmin = 0
        for j in range(index + 1):
            rmin += self.summary[j].middle
        
        return rmin
    
    def get_max_rank(self, index: int, rmin: int) -> int:
        """Get maximum rank at index."""
        delta = self.summary[index].right
        rmax = delta + rmin
        return rmax
    
    def quantile(self, phi: float) -> float:
        """Estimate a quantile for a given probability."""
        if phi < 0 or phi > 1.0:
            raise ValueError("quantile must be between 0 and 1")
        
        target_rank = int(math.ceil(phi * self.numx))
        rhs = self.epsilon * self.numx
        
        if target_rank >= self.numx:
            return self.xmax
        
        val = self.xmax
        for i in range(len(self.summary)):
            min_rank = self.get_min_rank(i)
            max_rank = self.get_max_rank(i, min_rank)
            
            if (target_rank - min_rank <= rhs) and (max_rank - target_rank <= rhs):
                val = self.summary[i].left
                break
        
        return val
    
    def get_epsilon(self) -> float:
        return self.epsilon
    
    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon
    
    def get_xmax(self) -> float:
        return self.xmax
    
    def get_numx(self) -> int:
        return self.numx
    
    def get_xmin(self) -> float:
        return self.xmin
    
    def size(self) -> int:
        """Returns sketch summary size."""
        return len(self.summary)
    
    def get_summary(self) -> List[Triple]:
        return self.summary
    
    def set_summary(self, summary: List[Triple]) -> None:
        self.summary = summary
    
    def approx_quantiles(self, probs: List[float]) -> List[float]:
        """Get approximate quantiles for given probabilities."""
        quantiles = []
        for phi in probs:
            q = self.quantile(phi)
            quantiles.append(q)
        
        return quantiles
    
    def __str__(self) -> str:
        return f"GKSketch [summary={self.summary}, xmin={self.xmin}, xmax={self.xmax}, numx={self.numx}]"