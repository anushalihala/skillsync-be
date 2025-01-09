import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class RAGAgent:
    def __init__(self, cv_file_path, job_desc):
        self.llm = OpenAI(model="gpt-4o-mini")
        Settings.llm = self.llm
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

        # load documents
        documents = SimpleDirectoryReader(input_files=[cv_file_path]).load_data()
        splitter = SentenceSplitter(chunk_size=1024)
        self.cv_nodes = splitter.get_nodes_from_documents(documents)

        documents = [Document(text=job_desc)]
        splitter = SentenceSplitter(chunk_size=1024)
        self.job_nodes = splitter.get_nodes_from_documents(documents)
        self.cv_vector_tool = None
        self.job_vector_tool = None
        self.cv_summary_tool = None
        self.job_summary_tool = None
        self._get_vector_tools()
        self._get_summary_tools()

    def _get_vector_tools(self):
        if self.cv_vector_tool is None:
            cv_vector_index = VectorStoreIndex(self.cv_nodes)
            cv_vector_query_engine = cv_vector_index.as_query_engine()
            self.cv_vector_tool = QueryEngineTool.from_defaults(
                name="cv_vector_tool",
                query_engine=cv_vector_query_engine,
                description=(
                    "Useful for retrieving specific context about user's CV/resume."
                ),
            )

        if self.job_vector_tool is None:
            job_vector_index = VectorStoreIndex(self.job_nodes)
            job_vector_query_engine = job_vector_index.as_query_engine()
            self.job_vector_tool = QueryEngineTool.from_defaults(
                name="job_vector_tool",
                query_engine=job_vector_query_engine,
                description=(
                    "Useful for retrieving specific context about job posting user is applying to."
                ),
            )

    def _get_summary_tools(self):
        if self.cv_summary_tool is None:
            cv_summary_index = SummaryIndex(self.cv_nodes)
            cv_summary_query_engine = cv_summary_index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True,
            )
            self.cv_summary_tool = QueryEngineTool.from_defaults(
                name="cv_summary_tool",
                query_engine=cv_summary_query_engine,
                description=(
                    "Useful for summarization questions related to user's CV/resume."
                ),
            )

        if self.job_summary_tool is None:
            job_summary_index = SummaryIndex(self.job_nodes)
            job_summary_query_engine = job_summary_index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True,
            )
            self.job_summary_tool = QueryEngineTool.from_defaults(
                name="job_summary_tool",
                query_engine=job_summary_query_engine,
                description=(
                    "Useful for summarization questions related to job posting user is applying to."
                ),
            )

    def _auto_rag(self, prompt):
        agent_worker = FunctionCallingAgentWorker.from_tools(
            [
                self.cv_vector_tool,
                self.job_vector_tool,
                self.cv_summary_tool,
                self.job_summary_tool,
            ],
            llm=self.llm,
            verbose=True,
        )
        agent = AgentRunner(agent_worker, verbose=True)
        response = agent.query(prompt)
        return str(response)

    def _guided_rag(self, beginning_prompt):
        job_vector_query_engine = self.job_vector_tool.query_engine
        hard_skills = job_vector_query_engine.query(
            "What are the hard skills and tool knowledge required for this position?"
        )
        soft_skills = job_vector_query_engine.query(
            "What are the soft skills required for this position?"
        )
        quals = job_vector_query_engine.query(
            "What are the qualifications required for this position?"
        )

        cv_vector_query_engine = self.cv_vector_tool.query_engine
        hard_skills_evid = cv_vector_query_engine.query(str(hard_skills))
        soft_skills_evid = cv_vector_query_engine.query(str(soft_skills))
        quals_evid = cv_vector_query_engine.query(str(quals))

        cvsum = self.cv_summary_tool.query_engine.query("cv summary")
        jobsum = self.job_summary_tool.query_engine.query("job summary")

        agent_worker = FunctionCallingAgentWorker.from_tools(
            [
                self.cv_vector_tool,
                self.job_vector_tool,
                self.cv_summary_tool,
                self.job_summary_tool,
            ],
            llm=self.llm,
            verbose=True,
        )
        agent = AgentRunner(agent_worker, verbose=True)
        response = agent.query(
            f"""{beginning_prompt}

        CV summary:
        {str(cvsum)}

        Job description summary:
        {str(jobsum)}

        Emphasize the following:
        {str(hard_skills_evid)}

        {str(soft_skills_evid)}

        {str(quals_evid)}

        Use the available tools for any other information you need.
        Specific questions about the CV can be answered using cv_vector_tool and specific questions about the job can be answered using job_vector_tool.
        """
        )
        return str(response)

    def rag_task(self, task_type: str):
        if task_type == "auto_cv":
            return self._auto_rag(
                "Tailor the user's CV (summary provided in cv_summary_tool and specific questions answered using cv_vector_tool) to be the best possible fit for the job description (summary provided in and specific questions answered using job_summary_tool and job_vector_tool). Do not invent information not in the user's CV."
            )
        elif task_type == "auto_cover_letter":
            return self._auto_rag(
                "Craft a cover letter using the user's resume (summary provided in cv_summary_tool and specific questions answered using cv_vector_tool) to demonstrate that the user is an ideal fit for the job (summary provided in job_summary_tool and specific questions answered using  and job_vector_tool). Do not invent information not in the user's CV."
            )
        elif task_type == "guided_cv":
            return self._guided_rag(
                "Tailor the user's CV to be the best possible fit for the job description. Do not invent information not in the user's CV."
            )
        elif task_type == "guided_cover_letter":
            return self._guided_rag(
                "Craft a cover letter using the user's CV to demonstrate that the user is an ideal fit for the job. Do not invent information not in the user's CV."
            )
        else:
            raise Exception("Invalid task type")
