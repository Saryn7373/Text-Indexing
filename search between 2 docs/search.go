package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

var cleanupRe = regexp.MustCompile(`[^\p{L}\p{N}\s]+`)

func loadStops(path string) (map[string]struct{}, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	stop := make(map[string]struct{})
	for _, w := range strings.Fields(string(data)) {
		stop[strings.ToLower(w)] = struct{}{}
	}

	return stop, nil
}

func tokenize(text string) []string {
	text = strings.ToLower(text)
	text = cleanupRe.ReplaceAllString(text, " ")
	return strings.Fields(text)
}

type WordFreq map[string]int

func buildWordFreq(text string, stop map[string]struct{}) WordFreq {
	res := make(WordFreq)
	for _, w := range tokenize(text) {
		if _, isStop := stop[w]; isStop {
			continue
		}
		res[w]++
	}

	return res
}

type Document struct {
	Id   string
	Text string
}

func loadDocs(dir string) ([]Document, error) {
	// stops, err := loadStops("stopwords-en.txt")
	// if err != nil {
	// 	return nil, err
	// }

	paths, err := filepath.Glob(filepath.Join(dir, "*.txt"))
	if err != nil {
		return nil, err
	}

	docs := make([]Document, 0, len(paths))
	for _, p := range paths {
		data, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}

		docs = append(docs, Document{
			Id:   filepath.Base(p),
			Text: string(data),
		})
	}

	return docs, nil
}

type InverseIndex map[string]map[string]int
type DirectIndex map[string]map[string]int

func buildIndexes(docs []Document, stops map[string]struct{}) (DirectIndex, InverseIndex) {

	direct := make(DirectIndex)
	inverse := make(InverseIndex)

	for _, doc := range docs {
		for _, w := range tokenize(doc.Text) {

			if _, ok := stops[w]; ok {
				continue
			}

			if direct[doc.Id] == nil {
				direct[doc.Id] = map[string]int{w: 0}
			}
			direct[doc.Id][w]++

			if inverse[w] == nil {
				inverse[w] = map[string]int{doc.Id: 0}
			}
			inverse[w][doc.Id]++
		}
	}

	return direct, inverse
}

func avg(sum, n int) float64 {
	if n == 0 {
		return 0
	}

	return float64(sum) / float64(n)
}

func classifyQuery(query string, topic1, topic2 InverseIndex) (string, int) {
	words := tokenize(query)

	var sum1, sum2, n1, n2 int
	for _, w := range words {
		if _, ok := topic1[w]; ok {
			for _, freq := range topic1[w] {
				sum1 += freq
				n1++
			}
		}

		if _, ok := topic2[w]; ok {
			for _, freq := range topic2[w] {
				sum2 += freq
				n2++
			}
		}
	}

	avg1, avg2 := avg(sum1, n1), avg(sum2, n2)

	if avg1 >= avg2 {
		return "World of Warcraft", int(avg1)
	}
	return "Marvel", int(avg2)
}

type DocScore struct {
	Doc   string
	Score int
}

func rankDocs(query string, inverse InverseIndex, top int) []DocScore {
	words := tokenize(query)
	score := make(map[string]int)

	for _, w := range words {
		if _, ok := inverse[w]; !ok {
			continue
		}

		for id, freq := range inverse[w] {
			score[id] += freq
		}
	}

	res := make([]DocScore, 0, len(score))

	for id, freq := range score {
		res = append(res, DocScore{Doc: id, Score: freq})
	}

	sort.Slice(res, func(s1, s2 int) bool {
		return res[s1].Score > res[s2].Score
	})

	if len(res) > top {
		return res[:top]
	}
	return res
}

type Topic struct {
	Name    string
	Docs    []Document
	Direct  DirectIndex
	Inverse InverseIndex
}

func loadTopic(name, dir string, stops map[string]struct{}) (Topic, error) {
	docs, err := loadDocs(dir)
	if err != nil {
		return Topic{}, err
	}

	direct, inverse := buildIndexes(docs, stops)

	return Topic{
		Name:    name,
		Docs:    docs,
		Direct:  direct,
		Inverse: inverse,
	}, nil
}

func main() {
	stops, err := loadStops("stopwords-en.txt")
	if err != nil {
		fmt.Println("stops didn't loaded", err)
		stops = map[string]struct{}{}
	}

	topic1, err := loadTopic("World of Warcraft", `src\World of Warcraft`, stops)
	if err != nil {
		fmt.Println("topic1 didn't loaded", err)
		return
	}

	topic2, err := loadTopic("Marvel", `src\Marvel`, stops)
	if err != nil {
		fmt.Println("topic2 didn't loaded", err)
		return
	}

	fmt.Printf("Загружено документов: %s — %d, %s — %d\n", topic1.Name, len(topic1.Docs), topic2.Name, len(topic2.Docs))
	fmt.Println("Введите поисковый запрос (0 — выход):")

	fmt.Println(topic1.Direct)

	// scanner := bufio.NewScanner(os.Stdin)
	// for {
	// 	fmt.Print("> ")
	// 	if !scanner.Scan() {
	// 		break // Ctrl+D / EOF
	// 	}

	// 	query := strings.TrimSpace(scanner.Text())
	// 	if query == "0" {
	// 		fmt.Println("Выход")
	// 		break
	// 	}
	// 	if query == "" {
	// 		continue
	// 	}

	// 	topicName, score := classifyQuery(query, topic1.Inverse, topic2.Inverse)
	// 	fmt.Printf("Тема: %s (score=%d)\n", topicName, score)

	// 	var chosen Topic
	// 	if topicName == topic1.Name {
	// 		chosen = topic1
	// 	} else {
	// 		chosen = topic2
	// 	}

	// 	top3 := rankDocs(query, chosen.Inverse, 3)
	// 	if len(top3) == 0 {
	// 		fmt.Println("No matching docs")
	// 	} else {
	// 		fmt.Println("Топ-3 документа:")
	// 		for i, ds := range top3 {
	// 			fmt.Printf("  %d. %s (score=%d)\n", i+1, ds.Doc, ds.Score)
	// 		}
	// 	}
	// 	fmt.Println()
	// }
}
