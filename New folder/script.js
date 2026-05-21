const form = document.getElementById("essay-form");
const output = document.getElementById("essay-output");
const statusLabel = document.getElementById("status");
const copyButton = document.getElementById("copy-btn");

const introTemplates = {
  formal: ({ topic, thesis }) =>
    `${topic} remains a significant subject in contemporary social discussion because it raises central questions about equality, justice, and participation in public life. Feminism, as both a body of ideas and a social movement, has consistently challenged systems that restrict women's rights and opportunities. ${thesis}`,
  academic: ({ topic, thesis }) =>
    `${topic} can be examined through the broader framework of feminist thought, which evaluates how gendered power structures shape institutions, expectations, and access to resources. Across historical periods, feminist movements have sought legal reform, cultural change, and greater social inclusion. ${thesis}`,
  simple: ({ topic, thesis }) =>
    `${topic} is important because feminism is about fairness and equal opportunity. It helps people question unfair rules and attitudes that limit women in education, work, politics, and daily life. ${thesis}`,
  persuasive: ({ topic, thesis }) =>
    `${topic} deserves serious attention because gender inequality is not just a private issue, but a public problem that affects families, schools, workplaces, and governments. Feminism gives society a way to recognize injustice and push for lasting change. ${thesis}`
};

const conclusionTemplates = {
  formal: ({ topic }) =>
    `In conclusion, ${topic.toLowerCase()} demonstrates why feminism continues to matter in modern society. By challenging discrimination, expanding opportunity, and demanding equal dignity, feminism contributes to a more just and balanced world for everyone.`,
  academic: ({ topic }) =>
    `In conclusion, ${topic.toLowerCase()} illustrates the enduring relevance of feminist analysis. The movement's contribution lies not only in legal reform, but also in its ability to reshape social values, institutional practices, and public understanding of equality.`,
  simple: ({ topic }) =>
    `In conclusion, ${topic.toLowerCase()} shows that feminism is still necessary today. It helps build a society where people are respected, heard, and given equal chances regardless of gender.`,
  persuasive: ({ topic }) =>
    `In conclusion, ${topic.toLowerCase()} makes it clear that feminism should be seen as a necessary force for progress rather than a divisive idea. Supporting gender equality strengthens communities, improves opportunity, and moves society closer to real justice.`
};

const angleContent = {
  equality: [
    "One of the main goals of feminism is to secure equal rights and equal treatment under the law.",
    "It challenges the belief that gender should determine a person's freedom, authority, or social value.",
    "By promoting fairness in education, politics, and family life, feminism helps create a more balanced society."
  ],
  history: [
    "The history of feminism shows that women's rights were won through long and difficult struggles rather than granted automatically.",
    "Early feminist movements focused on voting rights and legal recognition, while later movements addressed employment, safety, and representation.",
    "Understanding this development reveals how social change often depends on organized activism and public pressure."
  ],
  education: [
    "Feminism emphasizes that education is one of the strongest tools for social mobility and personal independence.",
    "When girls and women have equal access to education, they gain stronger voices in family decisions, employment, and civic life.",
    "This perspective also challenges cultural assumptions that restrict ambition or define intelligence through gender stereotypes."
  ],
  workplace: [
    "In the workplace, feminism draws attention to wage gaps, underrepresentation in leadership, and biased expectations about competence.",
    "It argues that professional advancement should depend on merit and opportunity rather than outdated assumptions about gender roles.",
    "This approach benefits institutions as well, because more inclusive workplaces tend to produce broader ideas and stronger decision-making."
  ],
  misconceptions: [
    "A common misconception is that feminism seeks female superiority, when in fact its central aim is equality.",
    "Another misunderstanding is that feminism is no longer needed, even though discrimination, harassment, and unequal access still persist.",
    "By correcting these myths, feminist thought becomes easier to understand as a human rights issue rather than a conflict between genders."
  ]
};

const detailByLength = {
  short: 2,
  medium: 3,
  long: 5
};

const supportingIdeas = {
  short: [
    "Feminism also helps question harmful stereotypes that limit both women and men.",
    "Its message encourages mutual respect rather than social division."
  ],
  medium: [
    "Feminist thinking also highlights how media, language, and cultural norms can normalize unequal expectations.",
    "It encourages people to examine not only major laws and policies, but also everyday practices that shape confidence, safety, and access.",
    "As a result, feminism operates at both the personal and institutional level."
  ],
  long: [
    "Feminist analysis also shows that inequality is often reinforced through ordinary routines rather than dramatic legal barriers alone.",
    "For example, assumptions about caregiving, leadership style, emotional behavior, and personal safety can quietly shape educational and professional outcomes.",
    "Modern feminist discussion therefore pays attention to language, media representation, domestic labor, and access to public space.",
    "This broader view helps explain why legal equality, although important, is not always enough by itself.",
    "Real change usually requires cultural change as well."
  ]
};

function thesisByTone(tone, angle) {
  const map = {
    equality: "It argues that a just society must ensure that rights and opportunities are not limited by gender.",
    history: "It shows that the gains women enjoy today are the result of organized struggle and continued public engagement.",
    education: "It demonstrates that equal access to learning and self-development is essential to social progress.",
    workplace: "It makes clear that genuine progress depends on equal opportunity, fair treatment, and representation in leadership.",
    misconceptions: "It reveals that many criticisms of feminism are based on misunderstanding rather than careful analysis."
  };

  if (tone === "persuasive") {
    return map[angle].replace("It ", "For that reason, feminism ");
  }

  return map[angle];
}

function buildParagraph(sentences) {
  return sentences.join(" ");
}

function generateEssay({ topic, tone, length, angle, includeCounterargument }) {
  const thesis = thesisByTone(tone, angle);
  const intro = introTemplates[tone]({ topic, thesis });
  const angleSentences = angleContent[angle].slice(0, detailByLength[length]);
  const supportSentences = supportingIdeas[length];

  const bodyOne = buildParagraph(angleSentences);
  const bodyTwo = buildParagraph(supportSentences);

  const bodyThree = buildParagraph([
    "Another important part of feminism is its focus on protecting women from discrimination and violence.",
    "Whether the issue is harassment, exclusion from decision-making, unequal pay, or pressure to conform to rigid social roles, feminism provides a language for identifying the problem and demanding accountability.",
    tone === "simple"
      ? "This makes the movement practical as well as meaningful."
      : "This makes the movement both intellectually significant and practically necessary."
  ]);

  const paragraphs = [intro, bodyOne, bodyTwo, bodyThree];

  if (includeCounterargument) {
    paragraphs.push(
      buildParagraph([
        "Some critics claim that feminism is no longer necessary or that it creates division between men and women.",
        "However, this view ignores the persistence of structural inequality and misunderstands feminism's actual purpose.",
        "The movement does not seek privilege for one group over another; it seeks a fair social order in which people are judged by their abilities, character, and choices rather than by restrictive gender expectations."
      ])
    );
  }

  paragraphs.push(conclusionTemplates[tone]({ topic }));

  return paragraphs.join("\n\n");
}

async function copyEssay() {
  const text = output.textContent.trim();

  if (!text || text.startsWith("Choose your settings")) {
    statusLabel.textContent = "Generate an essay first";
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    statusLabel.textContent = "Essay copied";
  } catch (error) {
    statusLabel.textContent = "Copy failed";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();

  const topic = document.getElementById("topic").value.trim() || "The role of feminism in society";
  const tone = document.getElementById("tone").value;
  const length = document.getElementById("length").value;
  const angle = document.getElementById("angle").value;
  const includeCounterargument = document.getElementById("include-counterargument").checked;

  output.textContent = generateEssay({
    topic,
    tone,
    length,
    angle,
    includeCounterargument
  });

  statusLabel.textContent = "Essay generated";
});

copyButton.addEventListener("click", copyEssay);
